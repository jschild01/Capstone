import os
import sys
import whisper
import torch
from tqdm import tqdm
import logging
import json
import argparse


def format_timestamp(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def write_txt_with_timecodes(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            f.write(f"[{start_time} --> {end_time}] {segment['text'].strip()}\n")


def write_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment['start']).replace('.', ',')
            end_time = format_timestamp(segment['end']).replace('.', ',')
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def transcribe_and_translate(model, mp3_path, output_formats, output_dir):
    try:
        # Transcribe in the original language
        result = model.transcribe(mp3_path)
        detected_language = result["language"]
        segments = result["segments"]

        base_name = os.path.splitext(os.path.basename(mp3_path))[0]

        outputs = []
        for format in output_formats:
            if format == "txt":
                output_path = os.path.join(output_dir, f"{base_name}_{detected_language}.txt")
                write_txt_with_timecodes(segments, output_path)
            elif format == "srt":
                output_path = os.path.join(output_dir, f"{base_name}_{detected_language}.srt")
                write_srt(segments, output_path)
            outputs.append(output_path)

        # If the detected language is not English, create an English translation
        if detected_language != "en":
            translation = model.transcribe(mp3_path, task="translate")
            translated_segments = translation["segments"]

            for format in output_formats:
                if format == "txt":
                    output_path = os.path.join(output_dir, f"{base_name}_{detected_language}_en_translation.txt")
                    write_txt_with_timecodes(translated_segments, output_path)
                elif format == "srt":
                    output_path = os.path.join(output_dir, f"{base_name}_{detected_language}_en_translation.srt")
                    write_srt(translated_segments, output_path)
                outputs.append(output_path)

        return f"Processed {os.path.basename(mp3_path)} (Detected language: {detected_language}). Outputs: {', '.join(outputs)}"
    except Exception as e:
        error_message = f"Error processing {os.path.basename(mp3_path)}: {str(e)}"
        logging.error(error_message)
        return error_message


def process_mp3s(folder_path, model_name, output_formats, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    print(f"Loading {model_name} model...")
    model = whisper.load_model(model_name).to(device)

    mp3_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp3')]

    total_files = len(mp3_files)
    print(f"Found {total_files} MP3 files to process")
    logging.info(f"Found {total_files} MP3 files to process")

    for mp3_path in tqdm(mp3_files, desc="Processing"):
        result = transcribe_and_translate(model, mp3_path, output_formats, output_dir)
        print(result)
        logging.info(result)

    print(f"Processing complete. Processed {total_files} files.")
    logging.info(f"Processing complete. Processed {total_files} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and translate MP3 files using Whisper")
    parser.add_argument("folder_path", help="Path to the folder containing MP3 files")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"],
                        default="large", help="Whisper model to use (default: large)")
    parser.add_argument("--output-formats", nargs='+',
                        choices=["txt", "srt"],
                        default=["txt", "srt"], help="Output formats for transcriptions (default: txt srt)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save transcriptions (default: same as input folder)")

    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"Error: {args.folder_path} is not a valid directory")
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else args.folder_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_mp3s(args.folder_path, args.model, args.output_formats, output_dir)