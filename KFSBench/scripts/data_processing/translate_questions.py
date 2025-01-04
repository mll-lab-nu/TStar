import json
from googletrans import Translator
import time

# Load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Save JSON file
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Translate questions to Chinese with retry logic
def translate_questions(json_data):
    translator = Translator()
    for entry in json_data:
        translated_questions = []
        for question in entry['questions']:
            success = False
            while not success:
                try:
                    translated = translator.translate(question, src='en', dest='zh-cn').text
                    translated_questions.append(translated)
                    success = True
                except AttributeError as e:
                    print(f"An error occurred: {e}")
                    # Retry after a short delay if an AttributeError occurs
                    time.sleep(1)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    translated_questions.append(question)  # Fallback to original question
                    success = True
        entry['questions'] = translated_questions
        print(translated_questions)
    return json_data

# Main function
def main():
    input_file_path = 'data/questions-all-formatted.json'
    output_file_path = 'data/questions-all-translated.json'
    
    # Load the original JSON
    json_data = load_json(input_file_path)
    
    # Translate questions to Chinese
    translated_data = translate_questions(json_data)
    
    # Save the translated JSON
    save_json(translated_data, output_file_path)
    print(f"Translated JSON saved to {output_file_path}")

if __name__ == "__main__":
    main()
