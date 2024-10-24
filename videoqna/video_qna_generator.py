import pandas as pd
import subprocess
import os
import sys
import csv
import re
import subprocess
import yt_dlp

from openai import OpenAI

from videoqna.ResponseSchemas import MCQCollection, SubjectiveCollection

def sanitize_filename(title):
    return re.sub(r'[^a-zA-Z0-9]', '_', title)

def download_media_from_youtube(youtube_url, ffmpeg_path="/usr/bin/ffmpeg"):
    # yt = YouTube(youtube_url)
    with yt_dlp.YoutubeDL() as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        video_title = sanitize_filename(ydl.sanitize_info(info)["fulltitle"])
        duration = ydl.sanitize_info(info)["duration_string"]
        print(info)
    download_path = f"./log/{video_title}"
    audio_output_file = os.path.join(download_path, f"{video_title}_audio.wav")

    # Download audio
    audio_command = [
        'yt-dlp',
        '-x',  # Extract audio only
        '--audio-format', 'wav',
        '--ffmpeg-location', ffmpeg_path,
        '-o', audio_output_file,
        youtube_url
    ]
    subprocess.run(audio_command, check=True)

    return audio_output_file, duration

def transcribe_audio_with_whisper(audio_file, model_size="base", language="English"):
    audio_path = os.path.dirname(audio_file)
    # Command to run Whisper for transcription
    command = [
        'whisper',
        audio_file,
        '--language', language,
        '--model', model_size,
        '--output_dir', audio_path,
        '--output_format', "srt"
    ]
    subprocess.run(command)
    
    print("Audio file: ", audio_file)
    print("Audio path: ", audio_path)
    srt_file_name = os.path.splitext(os.path.basename(audio_file))[0] + ".srt"
    srt_file_path = os.path.join(audio_path, srt_file_name)
    print("srt_path:", srt_file_path)
    return srt_file_path

def validate_srt(file_path):
    """
    Validates if the provided file path is an existing SRT file.
    """
    if not os.path.isfile(file_path):
        print(f"The file {file_path} does not exist.")
        sys.exit(1)
    if not file_path.lower().endswith('.srt'):
        print("The file is not a valid SRT file.")
        sys.exit(1)

def read_srt(file_path):
    """
    Reads the content of the SRT file, preserving its structure.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_txt(content, file_path):
    """
    Creates a new TXT file and writes the SRT content to it with the required formatting.
    """
    new_file_path = file_path.rsplit('.', 1)[0] + '.txt'
    with open(new_file_path, 'w', encoding='utf-8') as file:
        entry = []
        for line in content:
            if line.strip().isdigit():
                if entry:
                    file.write('\n'.join(entry) + '\n\n')
                    entry = []
                entry.append(line.strip())
            elif '-->' in line:
                entry.append(line.strip() + ' ')
            else:
                if line.strip():  # This avoids writing blank lines within subtitle entries
                    entry.append(line.strip())
        if entry:  # Write the last entry if the file doesn't end with a newline
            file.write('\n'.join(entry) + '\n')
    return new_file_path

def convert_srt_to_txt(file_path):
    """
    Converts an SRT file to a TXT file, handling errors gracefully.
    """
    try:
        validate_srt(file_path)
        content = read_srt(file_path)
        new_file_path = write_txt(content, file_path)
        print(f"Conversion successful. TXT file created at {new_file_path}")
        return new_file_path
    except Exception as e:
        print(f"An error occurred: {e}")

def read_transcript_from_file(file_path):
    """
    Reads the transcript text from a given file path.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        transcript = file.read()
    return transcript

def generate_learning_activities(transcript, output_file_path, question_format):
    """
    Sends a transcript to ChatGPT to generate ideas for fun learning activities.
    """
    prompt_text = transcript[:min(len(transcript), 8000)]  # Adjust based on your token budget

    if question_format == "subjective":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful chapter generator for video transcripts. Your task is to analyze the transcript content to generate chapters. For each identified chapter, generate a concise and descriptive chapter title or summary that captures the main topic or content of that chapter. Each chapter's starting timestamp must be from where the last one ended. Additionally, generate up to one question related to the content of each chapter to encourage critical thinking and understanding. Also, generate the answer to the question you will generate for each chapter. Ensure that each chapter detail is clearly separated and presented in a straightforward manner. Ensure that the discussion on each topic is finished and then generate the aforementioned things. Also, Ensure both the Question and the answer are short in length and concise and are evenly spaced out between topics. Segment topics into relevant chapters only. Remove milliseconds from the timestamps in output."
            },
            {
                "role": "user",
                "content": f"Based on the following transcript, generate chapter titles, descriptions, questions, answers and the requested information in the specified format:\n\n{prompt_text}"
            }
        ]

        api_key = os.getenv("API_KEY") #Enter your api key here
        client = OpenAI(api_key=api_key)
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=SubjectiveCollection
        )
        chapters_response = completion.choices[0].message
        if chapters_response.parsed:
            return chapters_response.parsed.model_dump()['subjective_questions']
        elif chapters_response.refusal:
            return chapters_response
        
    elif question_format == "mcq":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful chapter generator for video transcripts. Your task is to analyze the transcript content to generate chapters. For each identified chapter, generate a concise and descriptive chapter title or summary that captures the main topic or content of that chapter. Each chapter's beginning timestamp must start from where the last one ended. Additionally, generate up to one question related to the content of each chapter to encourage critical thinking and understanding. Also, generate fours options for the question you will generate for each chapter. Ensure that each chapter detail is clearly separated and presented in a straightforward manner. Ensure that the discussion on each topic is finished and then generate the aforementioned things. Also, Ensure both the Question and the options are short in length and concise and are evenly spaced out between topics. Segment topics into relevant chapters only. Remove milliseconds from the timestamps in output."
            },
            {
                "role": "user",
                "content": f"Based on the following transcript, generate chapter titles, descriptions, questions, answers and the requested information in the specified format:\n\n{prompt_text}"
            }
        ]
        client = OpenAI(api_key=os.getenv('API_KEY'))
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=MCQCollection
        )
        chapters_response = completion.choices[0].message
        if chapters_response.parsed:
            return chapters_response.parsed.model_dump()['mcq_sets']
        elif chapters_response.refusal:
            return chapters_response

def write_output_to_file(activities, output_file_path):
    """
    Writes the generated learning activities to a specified text file.

    Args:
    - activities (str): The generated activities to write.
    - output_file_path (str): The path of the output text file.
    """
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(activities)

def parse_chapter_info_from_file(input_file_path):
    # Read the contents of the file
    print(input_file_path)
    with open(input_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    print(text)
    # Regular expression to capture the relevant chapter details
    chapter_pattern = re.compile(
        r'Chapter No\. - (\d+)\s*'  # Capture chapter number
        r'Chapter Name - (.*?)\s*'  # Capture chapter name
        r'Chapter Start time - (.*?)\s*'  # Capture start time
        r'Chapter End Time - (.*?)\s*'  # Capture end time
        r'Chapter Question - (.*?)\s*'  # Capture question
        r'Chapter Answer - (.*?)\s*(?=Chapter No\. - \d+|$)',  # Capture answer, lookahead for next chapter or end of string
        re.DOTALL  # Dot matches newline as well
    )

    return chapter_pattern.findall(text)

def generate_video_qna(youtube_url, question_format):

    # Download Audio File
    print("Downloading media from YouTube...")
    audio_file, duration = download_media_from_youtube(youtube_url)
    print(f"Audio File: {audio_file}")

    # Transcribe Audio file to srt using whisper
    print("Transcribing audio with Whisper...")
    subtitle_file_path = transcribe_audio_with_whisper(audio_file)
    print("Transcription completed.")
    print(subtitle_file_path) 

    # convert .srt to .txt
    text_file_path = convert_srt_to_txt(subtitle_file_path)
    print("Text file saved at", text_file_path)

    transcript = read_transcript_from_file(text_file_path)
    input_dir = os.path.dirname(text_file_path)
    output_path = os.path.join(input_dir, "learning_activities.txt")
    
    #generate chapters
    chapters = generate_learning_activities(transcript, output_path, question_format)
    print(chapters)

    # chapters = [["1", "Introduction to Java", "00:00:00", "00:00:11", "Who designed Java and when?", "James Gosling designed Java in 1990."], ["2", "Java's Versatility and Use Cases", "00:00:11", "00:00:31", "What are some of the applications and systems powered by Java?", "Java powers enterprise web apps, big data pipelines with Hadoop, mobile apps on Android, and NASA's Maestro Mars Rover controller."], ["3", "Java's Compilation and Execution", "00:00:31", "00:00:48", "How does Java achieve platform independence?", "Java compiles to bytecode which can run on any operating system via the Java Virtual Machine (JVM)."], ["4", "Java's Language Features", "00:00:48", "00:01:09", "What high-level features does Java provide?", "Java provides garbage collection, runtime type checking, and reflection."], ["5", "Getting Started with Java", "00:01:09", "00:01:24", "What is required to start writing a Java program?", "Install the Java Development Kit (JDK) and create a file ending in .java with a class containing a main method."], ["6", "Java Syntax and Structure", "00:01:24", "00:01:59", "How do you define a variable and a method in Java?", "Define a variable with a type, name, and value. Define a method with the public and static keywords, a type, name, and return value."], ["7", "Compiling and Running Java Programs", "00:01:59", "00:02:10", "What are the steps to compile and run a Java program?", "Use the compiler to generate a .class file containing bytecode, then use the Java command to run it with the JVM."], ["8", "Conclusion and Call to Action", "00:02:10", "00:02:24", "What does the speaker promise if the video reaches 100,000 likes?", "The speaker promises to create a full Java tutorial."]]

    return chapters, duration

# def generate_video_qna(youtube_url):
#     return [["1", "Introduction to Java", "00:00:00", "00:00:11", "Who designed Java and when?", "James Gosling designed Java in 1990."], ["2", "Java's Versatility and Use Cases", "00:00:11", "00:00:31", "What are some of the applications and systems powered by Java?", "Java powers enterprise web apps, big data pipelines with Hadoop, mobile apps on Android, and NASA's Maestro Mars Rover controller."], ["3", "Java's Compilation and Execution", "00:00:31", "00:00:48", "How does Java achieve platform independence?", "Java compiles to bytecode which can run on any operating system via the Java Virtual Machine (JVM)."], ["4", "Java's Language Features", "00:00:48", "00:01:09", "What high-level features does Java provide?", "Java provides garbage collection, runtime type checking, and reflection."], ["5", "Getting Started with Java", "00:01:09", "00:01:24", "What is required to start writing a Java program?", "Install the Java Development Kit (JDK) and create a file ending in .java with a class containing a main method."], ["6", "Java Syntax and Structure", "00:01:24", "00:01:59", "How do you define a variable and a method in Java?", "Define a variable with a type, name, and value. Define a method with the public and static keywords, a type, name, and return value."], ["7", "Compiling and Running Java Programs", "00:01:59", "00:02:10", "What are the steps to compile and run a Java program?", "Use the compiler to generate a .class file containing bytecode, then use the Java command to run it with the JVM."], ["8", "Conclusion and Call to Action", "00:02:10", "00:02:24", "What does the speaker promise if the video reaches 100,000 likes?", "The speaker promises to create a full Java tutorial."]]
