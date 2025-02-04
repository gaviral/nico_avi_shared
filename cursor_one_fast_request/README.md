# Cursor One Fast Request

A tool to use in cursor to maximize the utilization of cursor fast requests, use speech to text and code more efficiently.


## How to use?

1. Have the openai api key setup in your ~/.zshrc file: `export OPENAI_API_KEY='<<your open ai api key>>'` (this is used for realtime speech recognition with openAI API. I used the script for ALOT! like 30-40 hours and that costed me ~$1.59) (you can also ask cursor to replace this with local whisper speeh to text or something along those lines)
2. Cursor's composer prompt: "Follow the instructions provided in the console logs of the assistant.py ".
3. Composer will the assistant.py speak into the mic when prompt. just say "that's it" for assistant to stop listening to your instructions and for composer to start following your instructions. just say "exit" to exit.
