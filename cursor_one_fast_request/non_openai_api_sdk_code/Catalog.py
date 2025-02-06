import subprocess
import time
import inspect
from typing import TypedDict, List

# Functions must be maximally granular
# Functions must strictly follow Single Responsibility Principle
# Functions must be ordered by dependency (dependent functions below their dependencies, functions that have longer dependency chains go further down, functions that have a lot of dependencies also go further down). When editing this file, feel free to rearrange the functions according to this rule.

class ScreenInfo(TypedDict):
    width: int
    height: int
    x: int
    y: int

class Catalog:
    # @staticmethod
    # def pause_execution(seconds: float) -> None:
    #     """Pauses execution for the specified number of seconds.
        
    #     Args:
    #         seconds: Number of seconds to pause execution
    #     """
    #     time.sleep(seconds)

    # @staticmethod
    # def execute_osascript_command(script: str) -> str:
    #     """Executes an AppleScript command and returns its output.
        
    #     Args:
    #         script: The AppleScript command to execute
            
    #     Returns:
    #         The stripped output of the command
    #     """
    #     result = subprocess.run(['osascript', '-e', script], 
    #                         capture_output=True, 
    #                         text=True, 
    #                         check=True)
    #     return result.stdout.strip()

    # @staticmethod
    # def handle_osascript_error(error: subprocess.CalledProcessError) -> str:
    #     """Handles errors that occur during AppleScript execution.
        
    #     Args:
    #         error: The CalledProcessError that occurred
            
    #     Returns:
    #         Empty string to indicate error
    #     """
    #     print(f"Error running AppleScript: {error}")
    #     return ""

    # @staticmethod
    # def run_apple_script(script: str) -> str:
    #     """Runs an AppleScript with error handling.
        
    #     Args:
    #         script: The AppleScript to run
            
    #     Returns:
    #         The output of the script, or empty string if an error occurred
    #     """
    #     try:
    #         return Catalog.execute_osascript_command(script)
    #     except subprocess.CalledProcessError as e:
    #         return Catalog.handle_osascript_error(e)

    # @staticmethod
    # def get_screens_information() -> List[ScreenInfo]:
    #     """Gets information about all connected screens.
        
    #     Returns:
    #         A list of dictionaries containing screen information.
    #         Each dictionary has keys:
    #             - 'width': Screen width in pixels
    #             - 'height': Screen height in pixels
    #             - 'x': Screen's x position
    #             - 'y': Screen's y position
    #     """
    #     script = '''
    #     tell application "System Events"
    #         set screenBounds to bounds of every window of desktop
    #         set screenInfo to {}
    #         repeat with bounds in screenBounds
    #             set end of screenInfo to {width:(item 3 of bounds) - (item 1 of bounds), height:(item 4 of bounds) - (item 2 of bounds), x:(item 1 of bounds), y:(item 2 of bounds)}
    #         end repeat
    #         return screenInfo
    #     end tell
    #     '''
    #     result = Catalog.run_apple_script(script)
    #     # Parse the AppleScript result into Python data structure
    #     # The result is in format {{width:w1, height:h1, x:x1, y:y1}, {width:w2, height:h2, x:x2, y:y2}, ...}
    #     screens: List[ScreenInfo] = []
    #     if result:
    #         # Remove outer braces and split by },{
    #         screen_strs = result.strip('{}').split('}, {')
    #         for screen_str in screen_strs:
    #             # Parse each screen's properties
    #             props: ScreenInfo = {'width': 0, 'height': 0, 'x': 0, 'y': 0}
    #             for prop in screen_str.split(', '):
    #                 key, value = prop.split(':')
    #                 props[key] = int(value)  # type: ignore
    #             screens.append(props)
    #     return screens

    # @staticmethod
    # def select_primary_screen() -> ScreenInfo:
    #     """Selects the primary screen, preferring the MacBook's built-in display.
        
    #     Returns:
    #         A dictionary containing the primary screen's information with keys:
    #             - 'width': Screen width in pixels
    #             - 'height': Screen height in pixels
    #             - 'x': Screen's x position
    #             - 'y': Screen's y position
    #     """
    #     screens = Catalog.get_screens_information()
    #     if not screens:
    #         # Fallback to default MacBook dimensions if no screens detected
    #         return {'width': 1440, 'height': 900, 'x': 0, 'y': 0}
        
    #     # Look for a screen with MacBook's common resolutions
    #     macbook_resolutions = [
    #         (1440, 900),  # MacBook Air 13" (pre-2018)
    #         (2560, 1600), # MacBook Pro 13" Retina
    #         (2880, 1800), # MacBook Pro 15" Retina
    #         (3024, 1964), # MacBook Pro 14" (2021+)
    #         (3456, 2234), # MacBook Pro 16" (2021+)
    #         (2304, 1440), # MacBook 12"
    #     ]
        
    #     for screen in screens:
    #         if (screen['width'], screen['height']) in macbook_resolutions:
    #             return screen
        
    #     # If no MacBook screen found, return the first screen
    #     return screens[0]

    # @staticmethod
    # def get_primary_screen_dimensions() -> "tuple[int, int]":
    #     """Gets the dimensions of the primary screen.
        
    #     Returns:
    #         A tuple of (width, height) in pixels
    #     """
    #     screen = Catalog.select_primary_screen()
    #     return (screen['width'], screen['height'])

    # @staticmethod
    # def get_screen_center(dimensions: "tuple[int, int]") -> "tuple[int, int]":
    #     """Calculates the center point of a screen with given dimensions.
        
    #     Args:
    #         dimensions: A tuple of (width, height) in pixels
            
    #     Returns:
    #         A tuple of (x, y) coordinates for the center point
    #     """
    #     return (dimensions[0] // 2, dimensions[1] // 2)

    # @staticmethod
    # def activate_application(app_name: str) -> str:
    #     """Activates (brings to front) a specific application.
        
    #     Args:
    #         app_name: Name of the application to activate
            
    #     Returns:
    #         The output of the AppleScript command
    #     """
    #     script = f'''
    #     tell application "{app_name}"
    #         activate
    #     end tell
    #     '''
    #     return Catalog.run_apple_script(script)

    # @staticmethod
    # def quit_application(app_name: str) -> str:
    #     """Quits (completely closes) a specific application.
        
    #     Args:
    #         app_name: Name of the application to quit
            
    #     Returns:
    #         The output of the AppleScript command
    #     """
    #     script = f'''
    #     tell application "{app_name}"
    #         quit
    #     end tell
    #     '''
    #     return Catalog.run_apple_script(script)

    # @staticmethod
    # def quit_terminal() -> str:
    #     """Quits (completely closes) the Terminal application.
        
    #     Returns:
    #         The output of the AppleScript command
    #     """
    #     return Catalog.quit_application("Terminal")

    # @staticmethod
    # def quit_firefox() -> str:
    #     """Quits (completely closes) the Firefox browser.
        
    #     Returns:
    #         The output of the AppleScript command
    #     """
    #     return Catalog.quit_application("Firefox")

    # @staticmethod
    # def open_google_chrome() -> None:
    #     """Opens Google Chrome and ensures at least one window is open."""
    #     script = '''
    #     tell application "Google Chrome"
    #         activate
    #         if (count of windows) = 0 then
    #             make new window
    #         end if
    #         set index of window 1 to 1
    #     end tell
    #     '''
    #     Catalog.run_apple_script(script)

    # @staticmethod
    # def open_firefox() -> None:
    #     """Opens Firefox and ensures at least one window is open."""
    #     script = '''
    #     tell application "Firefox"
    #         activate
    #         if (count of windows) = 0 then
    #             make new window
    #         end if
    #         set index of window 1 to 1
    #     end tell
    #     '''
    #     Catalog.run_apple_script(script)

    # @staticmethod
    # def open_cursor() -> None:
    #     """Opens and activates the Cursor IDE."""
    #     Catalog.activate_application("Cursor")

    # @staticmethod
    # def open_terminal() -> None:
    #     """Opens the Terminal application."""
    #     Catalog.activate_application("Terminal")

    # @staticmethod
    # def open_this_file_in_cursor(file_path: str) -> None:
    #     """Opens a specific file in Cursor IDE using AppleScript.
        
    #     Args:
    #         file_path: The path to the file to open
    #     """
    #     script = f'''
    #     tell application "Cursor"
    #         activate
    #         open "{file_path}"
    #     end tell
    #     '''
    #     Catalog.run_apple_script(script)

    # @staticmethod
    # def execute_keyboard_shortcut(keys: str, target_app: str = "") -> None:
    #     """Execute a single keyboard shortcut using AppleScript.
        
    #     Args:
    #         keys: The keyboard shortcut to execute (e.g., "command+enter")
    #              Format: modifier+key or just key
    #              Modifiers: command, option, control, shift
    #         target_app: Optional application to target with the shortcut.
    #                    If empty, shortcut is sent to the system.
    #     """
    #     # Convert the keys to AppleScript format
    #     key_parts = keys.lower().split('+')
    #     if len(key_parts) == 1:
    #         key = key_parts[0]
    #         script = f'''
    #         tell application "System Events"
    #             {f'tell application process "{target_app}"' if target_app else ""}
    #             keystroke "{key}"
    #             {f'end tell' if target_app else ""}
    #         end tell
    #         '''
    #     else:
    #         modifiers = key_parts[:-1]
    #         key = key_parts[-1]
    #         # Convert modifiers to AppleScript format
    #         modifier_map = {
    #             'command': 'command down',
    #             'option': 'option down',
    #             'control': 'control down',
    #             'shift': 'shift down'
    #         }
    #         modifier_str = ', '.join(modifier_map[mod] for mod in modifiers)
    #         script = f'''
    #         tell application "System Events"
    #             {f'tell application process "{target_app}"' if target_app else ""}
    #             keystroke "{key}" using {{{modifier_str}}}
    #             {f'end tell' if target_app else ""}
    #         end tell
    #         '''
    #     Catalog.run_apple_script(script)

    # @staticmethod
    # def execute_keyboard_shortcuts(shortcut_sequence: "list[str]", target_app: str = "") -> None:
    #     """Execute a sequence of keyboard shortcuts.
        
    #     Args:
    #         shortcut_sequence: List of keyboard shortcuts to execute in sequence
    #                          Each shortcut follows the format of execute_keyboard_shortcut
    #         target_app: Optional application to target with the shortcuts.
    #                    If empty, shortcuts are sent to the system.
    #     """
    #     for shortcut in shortcut_sequence:
    #         Catalog.execute_keyboard_shortcut(shortcut, target_app)
    #         # Add a small delay between shortcuts to ensure they're executed in order
    #         Catalog.pause_execution(0.1)

    # @staticmethod
    # def press_command_enter() -> None:
    #     """Execute the Command+Enter keyboard shortcut in Cursor."""
    #     Catalog.activate_application("Cursor")
    #     Catalog.pause_execution(0.1)  # Give time for Cursor to activate
    #     Catalog.execute_keyboard_shortcut("command+enter", "Cursor")

    # @staticmethod
    # def click_mouse_at_location(x: int, y: int) -> None:
    #     """Clicks the mouse at the specified screen coordinates.
        
    #     Args:
    #         x: The x coordinate to click at
    #         y: The y coordinate to click at
    #     """
    #     script = f'tell application "System Events" to click at {{{x}, {y}}}'
    #     Catalog.run_apple_script(script)

    # @staticmethod
    # def click_center_of_primary_screen() -> None:
    #     """Clicks at the center of the primary screen."""
    #     screen = Catalog.select_primary_screen()
    #     center_x = screen['x'] + (screen['width'] // 2)
    #     center_y = screen['y'] + (screen['height'] // 2)
    #     Catalog.click_mouse_at_location(center_x, center_y)

    # @staticmethod
    # def focus_chrome_tab_by_title(title_substring: str) -> None:
    #     """Brings focus to a Chrome tab that contains the given string in its title.
        
    #     Args:
    #         title_substring: String to search for in tab titles
    #     """
    #     script = f'''
    #     tell application "Google Chrome"
    #         activate
    #         set windowList to every window
    #         repeat with aWindow in windowList
    #             set tabList to every tab of aWindow
    #             repeat with aTab in tabList
    #                 if title of aTab contains "{title_substring}" then
    #                     set active tab index of aWindow to (get index of aTab)
    #                     set index of aWindow to 1
    #                     return
    #                 end if
    #             end repeat
    #         end repeat
    #     end tell
    #     '''
    #     Catalog.run_apple_script(script)

    # empty class for now
    pass

def get_catalog_functions() -> "list[str]":
    """Returns a list of all function names in the Catalog class.
    
    Returns:
        A list of function names as strings
    """
    # Get all members of the Catalog class
    members = inspect.getmembers(Catalog)
    # Filter for only functions (static methods are converted to functions)
    function_names = [name for name, member in members 
                     if callable(member) and not name.startswith('_')]
    # Print the function names
    print("Functions in Catalog class:")
    for name in function_names:
        print(f"- {name}")
    return function_names

def test_apple_script_trial() -> None:
    """Test function to verify basic AppleScript functionality."""
    print("Opening Chrome...")
    Catalog.open_google_chrome()
    Catalog.pause_execution(2)
    
    print("Opening Cursor...")
    Catalog.open_cursor()
    Catalog.pause_execution(2)
    
    print("Test complete!")
