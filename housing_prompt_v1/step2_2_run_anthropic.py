#!/usr/bin/env python3
import json
import time
import os
import sys
import argparse
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from anthropic import Anthropic

class BatchProcessingError(Exception):
    """Custom exception for batch processing errors"""
    pass

def setup_logging(log_file: str, verbose: bool) -> None:
    """Configure logging with both file and console output"""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def read_jsonl(file_path: str):
    """Read JSONL file and yield each line as a dictionary"""
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line in {file_path}: {str(e)}")
                    continue
    except IOError as e:
        raise BatchProcessingError(f"Error reading file {file_path}: {str(e)}")

def write_jsonl(file_path: str, data: Dict[str, Any], mode: str = 'a') -> None:
    """Write a dictionary to a JSONL file"""
    try:
        with open(file_path, mode) as file:
            file.write(json.dumps(data) + '\n')
    except IOError as e:
        raise BatchProcessingError(f"Error writing to file {file_path}: {str(e)}")

def get_completed_items(output_file: str) -> Set[str]:
    """Get set of custom_ids that have already been processed successfully"""
    completed = set()
    if os.path.exists(output_file):
        for item in read_jsonl(output_file):
            if 'custom_id' in item and not item.get('error'):
                completed.add(item['custom_id'])
    return completed

def process_batch_file(
    input_file: str,
    client: Anthropic,
    output_file: Optional[str] = None,
    replace: bool = False,
    continue_on_error: bool = True,
    dry_run: bool = False,
    sleep: float = 0,
    batch_size: int = 10000
) -> None:
    """
    Process a batch file with resume functionality, using batch API

    Args:
        input_file: Path to input JSONL file
        client: Initialized Anthropic client
        output_file: Path to output file
        replace: Whether to replace existing output file
        continue_on_error: Whether to continue processing on non-fatal errors
        dry_run: If True, only show what would be processed
        sleep: Seconds to sleep between requests
        batch_size: Number of requests per batch (max 10000)
    """
    if not input_file.endswith(".jsonl"):
        raise BatchProcessingError("Input file must be a JSONL file")

    if output_file is None:
        input_dir, input_filename = os.path.split(input_file)
        input_filename = input_filename[:-6] + "_output.jsonl"
        output_file = os.path.join(input_dir, "batch_" + input_filename)

    # Get already completed items if not replacing
    completed_items = set()
    if not replace and os.path.exists(output_file):
        completed_items = get_completed_items(output_file)
        logging.info(f"Found {len(completed_items)} previously completed items")

    # Read all input items
    input_data = list(read_jsonl(input_file))
    total_items = len(input_data)
    remaining_items = [item for item in input_data
                       if item.get('custom_id') not in completed_items]

    logging.info(f"Total items: {total_items}")
    logging.info(f"Remaining items: {len(remaining_items)}")

    if dry_run:
        logging.info("Dry run completed")
        return

    # Create/clear output file if replacing
    if replace and os.path.exists(output_file):
        mode = 'w'
        logging.info(f"Replacing existing output file: {output_file}")
    else:
        mode = 'a'
        logging.info(f"Appending to output file: {output_file}")

    # Write header if creating new file
    if mode == 'w':
        header = {
            "header": f"Output for {input_file}",
            "file": input_file,
            "timestamp": datetime.utcnow().isoformat(),
            "total_items": total_items
        }
        write_jsonl(output_file, header, mode='w')

    # Create batches of batch_size
    batches = [remaining_items[i:i + batch_size] for i in range(0, len(remaining_items), batch_size)]

    # Load or initialize batch log
    batch_log_file = output_file + ".batch_log.jsonl"
    existing_batches = {}
    if os.path.exists(batch_log_file):
        # Load existing batch log
        for entry in read_jsonl(batch_log_file):
            batch_id = entry.get("batch_id")
            if batch_id:
                existing_batches[batch_id] = entry
    else:
        # Create new batch log file
        with open(batch_log_file, 'w') as f:
            pass  # Just create the file

    for batch_num, batch_items in enumerate(batches):
        # Check if this batch has been submitted before
        batch_id = None
        batch_key = f"{input_file}_batch_{batch_num}"
        # See if we have an entry with this batch_key
        for batch_entry in existing_batches.values():
            if batch_entry.get("batch_key") == batch_key:
                batch_id = batch_entry.get("batch_id")
                break

        if batch_id:
            logging.info(f"Found existing batch ID {batch_id} for batch {batch_num}")
        else:
            # Prepare batch requests
            batch_requests = []
            for item in batch_items:
                custom_id = item.get("custom_id")
                body = item.get("body")

                if body is None:
                    logging.error(f"Missing 'body' in input item {custom_id}")
                    continue

                messages = body.get("messages")
                if not messages:
                    logging.error(f"Missing or empty 'messages' in body of item {custom_id}")
                    continue

                model_name = body.get("model")
                if not model_name:
                    logging.error(f"Missing 'model' in body of item {custom_id}")
                    continue

                temperature = body.get("temperature", 1.0)
                max_tokens = body.get("max_tokens", 4096)

                # Flatten system messages into a single system prompt, if any
                system_prompts = " ".join([msg["content"] for msg in messages if msg["role"] == "system"])
                user_messages = [msg for msg in messages if msg["role"] != "system"]

                if system_prompts:
                    # Prepend the system prompt to the first user message
                    if user_messages:
                        user_messages[0]["content"] = f"{system_prompts}\n\n{user_messages[0]['content']}"
                    else:
                        # If there are no user messages, create one
                        user_messages = [{"role": "user", "content": system_prompts}]

                batch_requests.append({
                    "custom_id": custom_id,
                    "params": {
                        "model": model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "messages": user_messages
                    }
                })

            # Submit the batch
            try:
                response = client.beta.messages.batches.create(
                    requests=batch_requests
                )
                batch_id = response.id
                logging.info(f"Submitted batch {batch_num} with batch ID {batch_id}")

                # Store batch ID and description in batch log file
                batch_entry = {
                    "batch_id": batch_id,
                    "batch_key": batch_key,
                    "batch_num": batch_num,
                    "submitted_at": datetime.utcnow().isoformat(),
                    "status": response.processing_status
                }
                write_jsonl(batch_log_file, batch_entry)
                existing_batches[batch_id] = batch_entry

            except Exception as e:
                logging.error(f"Error submitting batch {batch_num}: {str(e)}")
                if not continue_on_error:
                    raise
                else:
                    continue

        # Now, check the status of the batch
        try:
            batch_status = client.beta.messages.batches.retrieve(batch_id)
            batch_processing_status = batch_status.processing_status
            logging.info(f"Batch {batch_id} status: {batch_processing_status}")

            if batch_processing_status == "ended":
                # Retrieve results
                logging.info(f"Retrieving results for batch {batch_id}")
                results = client.beta.messages.batches.results(batch_id)

                for result in results:
                    custom_id = result.custom_id
                    if result.result.type == "succeeded":
                        response = result.result.message

                        output_item = {
                            "id": response.id,
                            "custom_id": custom_id,
                            "response": {
                                "status_code": 200,
                                "body": {
                                    "id": response.id,
                                    "object": "chat.completion",
                                    "created": int(time.time()),
                                    "model": response.model,
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": response.role,
                                            "content": response.content
                                        },
                                        "finish_reason": response.stop_reason,
                                    }],
                                    "usage": {
                                        "prompt_tokens": response.metrics.input_tokens,
                                        "completion_tokens": response.metrics.output_tokens,
                                        "total_tokens": response.metrics.input_tokens + response.metrics.output_tokens
                                    }
                                }
                            },
                            "error": None
                        }
                        write_jsonl(output_file, output_item, mode='a')
                        logging.debug(f"Successfully processed item {custom_id}")
                    else:
                        # Handle errors or other statuses
                        error = result.result
                        error_response = {
                            "id": None,
                            "custom_id": custom_id,
                            "response": None,
                            "error": {
                                "status_code": 500,
                                "type": error.type,
                                "message": error.error,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                        write_jsonl(output_file, error_response, mode='a')
                        logging.error(f"Error processing item {custom_id}: {error.error}")

                # Update batch entry status
                existing_batches[batch_id]['status'] = 'ended'
                write_jsonl(batch_log_file, existing_batches[batch_id])

            else:
                logging.info(f"Batch {batch_id} is still processing")
                # Update batch entry status
                existing_batches[batch_id]['status'] = batch_processing_status
                write_jsonl(batch_log_file, existing_batches[batch_id])

        except Exception as e:
            logging.error(f"Error retrieving results for batch {batch_id}: {str(e)}")
            if not continue_on_error:
                raise
            else:
                continue

        time.sleep(sleep)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process batch requests for Anthropic's Claude API with resume functionality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "-i", "--input-file",
        type=str,
        required=True,
        help="Input JSONL file path"
    )

    # Optional arguments
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        help="Output JSONL file path (default: batch_<input>_output.jsonl)"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing output file instead of resuming"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Anthropic API key (default: from ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing if individual items fail"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop processing if any item fails"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making API calls"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "-s", "--sleep",
        type=float,
        default=0,
        help="Time to sleep between requests"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of requests per batch (max 10000)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Setup logging
    log_file = f"{args.input_file}.log"
    setup_logging(log_file, args.verbose)

    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise BatchProcessingError("No API key provided. Use --api-key or set ANTHROPIC_API_KEY environment variable")

    # Initialize client
    client = Anthropic(api_key=api_key)

    try:
        process_batch_file(
            input_file=args.input_file,
            client=client,
            output_file=args.output_file,
            replace=args.replace,
            continue_on_error=not args.stop_on_error,
            dry_run=args.dry_run,
            sleep=args.sleep,
            batch_size=args.batch_size
        )
    except BatchProcessingError as e:
        logging.error(f"Batch processing failed: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.exception("Unexpected error occurred")
        sys.exit(1)
    else:
        logging.info("Batch processing completed successfully")

if __name__ == "__main__":
    main()
