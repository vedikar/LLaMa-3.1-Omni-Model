import os

# Define the path to the dataset
audio_dir = r"C:\Users\Vedika\.cache\LibriSpeech\train-clean-100"  # Path to your dataset directory

# Function to load transcription files (assuming transcription files are in .txt format)
def load_transcription(audio_file):
    # Get the base name of the audio file without the sequence number
    base_name = audio_file.rsplit('-', 1)[0]  # Split and discard the last part (sequence number)
    
    # Search recursively for the transcription file in the directory structure
    for root, dirs, files in os.walk(audio_dir):
        # Construct the transcription file name (e.g., "103-1240.trans.txt")
        text_file = f"{base_name}.trans.txt"
        if text_file in files:
            text_path = os.path.join(root, text_file)
            
            # Log the file path to check
            print(f"Found transcription file: {text_path}")
            
            try:
                with open(text_path, 'r') as f:
                    transcription = f.read().strip()  # Read the transcription
                print(f"Loaded transcription: {transcription[:50]}...")  # Print a snippet of the transcription
                return transcription
            except Exception as e:
                print(f"Error reading transcription file: {e}")
                return None
    print(f"Warning: Transcription file not found for {audio_file}")
    return None  # or an empty string if desired

# Function to search for all audio files and their corresponding transcriptions
def process_audio_files():
    for root, dirs, files in os.walk(audio_dir):
        for audio_file in files:
            if audio_file.endswith('.flac'):  # Check for .flac files
                print(f"Processing audio file: {audio_file}")
                transcription = load_transcription(audio_file)
                if transcription:
                    print(f"Transcription for {audio_file}: {transcription}")
                else:
                    print(f"No transcription found for {audio_file}")

# Run the script to process the audio files
process_audio_files()