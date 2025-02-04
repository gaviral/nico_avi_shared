from __future__ import annotations
import asyncio
import base64
import os
import sounddevice as sd  # type: ignore
import numpy as np
from non_openai_api_sdk_code.Catalog import Catalog
from openai import AsyncOpenAI
from dotenv import load_dotenv
import sys
import queue
import signal
import argparse
from typing import Optional, Dict, Any, Tuple, TypeVar, cast, List, get_type_hints
from dataclasses import dataclass
from time import time
import datetime
import inspect
import json

T = TypeVar('T')

def parse_command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Real-time audio transcription and conversation')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--show-assistant', action='store_true', help='Show assistant responses (hidden by default)')
    return parser.parse_args()

def print_to_console(*print_args: Any, **kwargs: Any) -> None:
    """Print message to console."""
    print(*print_args, **kwargs)

def log_to_file(message: str) -> None:
    """Log message to file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("non_openai_api_sdk_code/logs.md", "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def log_print(*print_args: Any, **kwargs: Any) -> None:
    """Print message to console and log to file."""
    print_to_console(*print_args, **kwargs)
    message = " ".join(str(arg) for arg in print_args)
    log_to_file(message)

def debug_print(*print_args: Any, **kwargs: Any) -> None:
    if CONFIG.debug:
        log_print(*print_args, **kwargs)

def validate_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        log_print("Error: OPENAI_API_KEY environment variable not found!")
        log_print("Please set your OpenAI API key in the .env file or environment variables.")
        sys.exit(1)

def setup_signal_handler() -> None:
    def handle_interrupt(signum: int, frame: Any) -> None:
        global running
        log_print("\nStopping...")
        running = False
    
    signal.signal(signal.SIGINT, handle_interrupt)

def create_audio_stream(audio_streamer: 'AudioStreamer') -> sd.InputStream:
    return sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=audio_streamer.callback,
        dtype=np.int16,
        blocksize=CHUNK_SIZE
    )

def encode_audio_to_base64(audio_data: np.ndarray[Any, np.dtype[np.int16]]) -> str:
    audio_bytes = audio_data.tobytes()
    return base64.b64encode(audio_bytes).decode('utf-8')

def get_session_config() -> Dict[str, Any]:
    return {
        "modalities": ["text"],
        "input_audio_format": "pcm16",
        "input_audio_transcription": {
            "model": "whisper-1"
        },
        "temperature": 0.7,
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "silence_duration_ms": 500,
            "prefix_padding_ms": 300
        },
        "instructions": "You are a helpful assistant that can control the computer using various functions. You can open applications, click on the screen, type keyboard shortcuts, and more. When a user asks you to do something, try to use the available functions to help them.",
        "tools": get_catalog_tools_config(),
        "tool_choice": "auto"
    }

def handle_thats_it_exit(transcript: str) -> bool:
    if "that's it" in transcript.lower():
        log_print("\nFollow exhautively all the instructions provided in the console log. If you have successfully followed all the instructions and the if you wrote code and that code works as expected then conventional commit + push it otherwise continue fixing the issues until they are all clearly fixed. in either case, don't stop working and run the assistant again. You will get the next instruction in the console logs of the next run.")
        return True
    return False

def handle_quick_exit(transcript: str) -> bool:
    if "exit" in transcript.lower():
        log_print("\nQuick exit requested. Goodbye!")
        return True
    return False

def check_exit_conditions(transcript: str) -> bool:
    return handle_thats_it_exit(transcript) or handle_quick_exit(transcript)

@dataclass
class Message:
    timestamp: float
    content: str
    is_complete: bool = False

class AudioStreamer:
    def __init__(self) -> None:
        self.audio_queue: queue.Queue[np.ndarray[Any, np.dtype[np.int16]]] = queue.Queue()
    
    def callback(self, indata: np.ndarray[Any, np.dtype[np.int16]], frames: int, time: Any, status: Any) -> None:
        if status:
            log_print(f"Status: {status}")
        self.audio_queue.put(indata.copy())

class ConversationManager:
    def __init__(self) -> None:
        self.messages: Dict[str, Message] = {}
        self.current_user_message_id: Optional[str] = None
        self.current_assistant_message_id: Optional[str] = None
        self.last_print_timestamp = 0
        self.assistant_prefix_printed = False
        self.current_assistant_response = ""
        self.has_active_response = False
        self.pending_messages: List[Tuple[str, str]] = []
    
    def add_user_message(self, message_id: str, transcript: str) -> None:
        """Add a user message to the conversation."""
        if self.has_active_response:
            # If there's an active response, queue this message
            self.pending_messages.append((message_id, transcript))
            log_print(f"Message queued: {transcript}")
            return
        
        self.messages[message_id] = Message(time(), transcript)
        self.current_user_message_id = message_id
        self.assistant_prefix_printed = False
        self.current_assistant_response = ""
        
        log_print(f"You: {transcript}\n")
        self.last_print_timestamp = time()
    
    def add_assistant_response(self, message_id: str, delta: str) -> None:
        if not CONFIG.show_assistant:
            return
        
        current_time = time()
        if message_id not in self.messages:
            self.messages[message_id] = Message(current_time, "")
            self.current_assistant_message_id = message_id
        
        if not self.assistant_prefix_printed:
            log_print("\nAssistant:", end=" ", flush=True)
            self.assistant_prefix_printed = True
        
        self.messages[message_id].content += delta
        self.current_assistant_response += delta
        log_print(delta, end="", flush=True)
    
    def complete_message(self, message_id: str) -> None:
        if message_id in self.messages:
            self.messages[message_id].is_complete = True
            if message_id == self.current_assistant_message_id and CONFIG.show_assistant:
                log_print("\n")
    
    def mark_response_started(self) -> None:
        """Mark that a response has started."""
        self.has_active_response = True
    
    def mark_response_complete(self) -> None:
        """Mark that a response is complete and process any pending messages."""
        self.has_active_response = False
        if self.pending_messages:
            # Process the next pending message
            message_id, transcript = self.pending_messages.pop(0)
            self.add_user_message(message_id, transcript)

async def process_audio_data(connection: Any, audio_streamer: AudioStreamer) -> Tuple[bool, Optional[str]]:
    try:
        audio_data = audio_streamer.audio_queue.get_nowait()
        encoded_audio = encode_audio_to_base64(audio_data)
        await connection.input_audio_buffer.append(audio=encoded_audio)
        return True, None
    except queue.Empty:
        return False, None
    except Exception as e:
        return False, str(e)

async def handle_transcription_event(event: Any, conversation: ConversationManager) -> Tuple[Optional[str], bool]:
    """Handle transcription completed event."""
    debug_print(f"DEBUG: Got transcription: {event.transcript}")
    conversation.add_user_message(event.item_id, event.transcript)
    if check_exit_conditions(event.transcript):
        return None, False
    if not conversation.has_active_response:
        conversation.mark_response_started()
    return event.item_id, True

async def handle_text_delta_event(event: Any, conversation: ConversationManager, current_message_id: Optional[str]) -> Tuple[Optional[str], bool]:
    """Handle text delta event."""
    debug_print(f"DEBUG: Got assistant delta: {event.delta}")
    if current_message_id is not None:
        conversation.add_assistant_response(current_message_id, event.delta)
    return current_message_id, True

async def handle_text_done_event(event: Any, conversation: ConversationManager, current_message_id: Optional[str]) -> Tuple[Optional[str], bool]:
    """Handle text done event."""
    debug_print("DEBUG: Assistant response complete")
    if current_message_id:
        conversation.complete_message(current_message_id)
    return current_message_id, True

async def handle_function_call(output_item: Any, connection: Any) -> None:
    """Handle a single function call."""
    debug_print(f"DEBUG: Function call: {output_item.name}")
    try:
        # Get the function from Catalog
        func = getattr(Catalog, output_item.name)
        # Parse arguments from JSON string
        args = json.loads(output_item.arguments)
        # Call the function
        result = func(**args)
        # Send function output back
        await connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": output_item.call_id,
                "output": json.dumps({"result": str(result) if result is not None else None})
            }
        )
        # Add a small delay between function calls to ensure proper execution order
        await asyncio.sleep(0.5)
    except Exception as e:
        debug_print(f"DEBUG: Function call error: {e}")
        # Send error back to model
        await connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": output_item.call_id,
                "output": json.dumps({"error": str(e)})
            }
        )

async def handle_response_done_event(event: Any, connection: Any, current_message_id: Optional[str], conversation: ConversationManager) -> Tuple[Optional[str], bool]:
    """Handle response done event."""
    debug_print("DEBUG: Response complete")
    if hasattr(event, 'response') and hasattr(event.response, 'output'):
        # Process all function calls in sequence
        for output_item in event.response.output:
            if output_item.type == "function_call":
                await handle_function_call(output_item, connection)
    
    # Mark response as complete and process any pending messages
    conversation.mark_response_complete()
    
    # If there are pending messages, they will be processed in mark_response_complete
    # Only create a new response if we have pending messages
    if conversation.pending_messages:
        await connection.response.create()
    
    return current_message_id, True

async def handle_event(event: Any, conversation: ConversationManager, current_message_id: Optional[str], connection: Any) -> Tuple[Optional[str], bool]:
    """Handle different types of events."""
    debug_print(f"\nDEBUG: Received event type: {event.type}")
    debug_print(f"DEBUG: Event details: {event}")
    
    if event.type == "conversation.item.input_audio_transcription.completed":
        return await handle_transcription_event(event, conversation)
    
    elif event.type == "response.text.delta":
        return await handle_text_delta_event(event, conversation, current_message_id)
    
    elif event.type == "response.text.done":
        return await handle_text_done_event(event, conversation, current_message_id)
    
    elif event.type == "response.done":
        return await handle_response_done_event(event, connection, current_message_id, conversation)
    
    elif event.type == "error":
        log_print(f"\nError: {event.error.message}")
        return current_message_id, True
    
    return current_message_id, True

async def process_single_event(connection: Any, timeout: float = 0.01) -> Optional[Any]:
    """Process a single event with timeout."""
    try:
        return await asyncio.wait_for(connection.recv(), timeout=timeout)
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        log_print(f"\nError receiving event: {e}")
        return None

async def handle_event_with_error_handling(event: Any, conversation: ConversationManager, current_message_id: Optional[str], connection: Any) -> Tuple[Optional[str], bool]:
    """Handle event with error handling."""
    try:
        return await handle_event(event, conversation, current_message_id, connection)
    except Exception as e:
        log_print(f"\nError handling event: {e}")
        return current_message_id, True

async def process_events(connection: Any, conversation: ConversationManager, current_message_id: Optional[str]) -> Tuple[Optional[str], bool]:
    """Process all available events."""
    try:
        while True:
            event = await process_single_event(connection)
            if event is None:
                break
            
            new_message_id, should_continue = await handle_event_with_error_handling(
                event, conversation, current_message_id, connection
            )
            
            if not should_continue:
                return new_message_id, False
            current_message_id = new_message_id
        
        return current_message_id, True
    except Exception as e:
        log_print(f"\nError processing events: {e}")
        return current_message_id, True

async def process_audio_stream(connection: Any, audio_streamer: AudioStreamer) -> bool:
    """Process audio stream and handle errors."""
    success, error = await process_audio_data(connection, audio_streamer)
    if not success:
        if error:
            log_print(f"\nError processing audio: {error}")
        await asyncio.sleep(0.01)
    return success

async def manage_conversation_state(connection: Any, conversation: ConversationManager, current_message_id: Optional[str]) -> Tuple[Optional[str], bool]:
    """Manage conversation state and process events."""
    current_message_id, should_continue = await process_events(
        connection, conversation, current_message_id
    )
    return current_message_id, should_continue

async def run_conversation_loop(connection: Any, audio_streamer: AudioStreamer, conversation: ConversationManager) -> None:
    """Run the main conversation loop."""
    current_message_id = None
    
    with create_audio_stream(audio_streamer) as stream:
        stream.start()
        
        while running:
            success = await process_audio_stream(connection, audio_streamer)
            if not success:
                continue
            
            current_message_id, should_continue = await manage_conversation_state(
                connection, conversation, current_message_id
            )
            if not should_continue:
                break

async def main() -> None:
    # Call temporary_testing at the beginning
    temporary_testing()

    client = AsyncOpenAI()
    audio_streamer = AudioStreamer()
    conversation = ConversationManager()
    
    log_print("Connecting to OpenAI Realtime API...")
    try:
        async with client.beta.realtime.connect(model="gpt-4o-mini-realtime-preview") as connection:
            log_print("Connected! Setting up session...")
            await connection.session.update(session=cast(Any, get_session_config()))
            
            log_print("\nListening... (Press Ctrl+C to stop)")
            log_print("-" * 50)
            
            await run_conversation_loop(connection, audio_streamer, conversation)
    
    except Exception as e:
        log_print(f"An error occurred: {e}")
    
    finally:
        log_print("\nStopped listening.")
        log_print("-" * 50)

# Audio configuration
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_DURATION = 0.05
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Global flag for running state
running = True

def temporary_testing():
    """Temporary function for testing functionality."""
    pass

def get_parameter_type_schema(param_type: Any) -> Dict[str, Any]:
    """Convert Python type to JSON schema type."""
    if param_type == str:
        return {"type": "string"}
    elif param_type == int:
        return {"type": "integer"}
    elif param_type == float:
        return {"type": "number"}
    elif param_type == bool:
        return {"type": "boolean"}
    elif str(param_type).startswith("typing.List"):
        return {"type": "array"}
    elif str(param_type).startswith("typing.Tuple"):
        return {"type": "array"}
    return {"type": "string"}  # fallback

def create_parameter_schema(sig: inspect.Signature, type_hints: Dict[str, Any]) -> Dict[str, Any]:
    """Create JSON schema for function parameters."""
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, None)
        parameters["properties"][param_name] = get_parameter_type_schema(param_type)
        
        if param.default == param.empty:
            parameters["required"].append(param_name)
    
    return parameters

def create_tool_definition(name: str, member: Any) -> Dict[str, Any]:
    """Create tool definition for a single function."""
    sig = inspect.signature(member)
    doc = inspect.getdoc(member) or ""
    type_hints = get_type_hints(member)
    
    parameters = create_parameter_schema(sig, type_hints)
    
    return {
        "type": "function",
        "name": name,
        "description": doc,
        "parameters": parameters
    }

def get_catalog_tools_config() -> List[Dict[str, Any]]:
    """Generates tools configuration for all Catalog functions."""
    tools: List[Dict[str, Any]] = []
    
    # Get all members of the Catalog class
    members = inspect.getmembers(Catalog)
    
    # Filter for static methods
    for name, member in members:
        if callable(member) and not name.startswith('_'):
            tool = create_tool_definition(name, member)
            tools.append(tool)
    
    return tools

if __name__ == "__main__":
    global CONFIG
    CONFIG = parse_command_line_args()
    load_dotenv()
    validate_api_key()
    setup_signal_handler()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log_print("\nProgram terminated by user")
    except Exception as e:
        log_print(f"Program terminated due to error: {e}") 
