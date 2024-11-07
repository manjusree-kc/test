import fitz  # PyMuPDF
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
import time
nltk.download('punkt')

import numpy as np  # Import NumPy for NaN values
import os# Set the GROQ_API_KEY directly in the environment
os.environ['GROQ_API_KEY'] = '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'
import os

from groq import Groq
import fitz  # PyMuPDF
from collections import defaultdict, Counter
import csv

#above is old version below is new version

def delete_pdf_file(file_path):
    """
    Deletes the PDF file at the specified file path.

    Args:
    - file_path (str): Path to the PDF file that needs to be deleted.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"The file '{file_path}' has been deleted.")
        else:
            print(f"The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"Error occurred while deleting the file: {e}")

# Step 1: Extract common y-coordinates (for headers/footers) and text blocks
def extract_common_y_coords(pdf_path):
    doc = fitz.open(pdf_path)

    # Check if the PDF has more than one page
    if len(doc) <= 1:
        print("PDF contains only one page. No processing needed.")
        doc.close()
        return None

    y_coords_all_pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        y_coords = [block[1] for block in blocks]  # Get y-coordinates
        y_coords_all_pages.append(set(y_coords))   # Store as set for easier intersection

    # Find common y-coordinates across all pages
    common_y_coords = set.intersection(*y_coords_all_pages)

    doc.close()
    return common_y_coords

# Step 2: Remove text from pages based on common y-coordinates
def remove_header_footer(pdf_path, common_y_coords, tolerance=10):
    if common_y_coords is None:
        print("Skipping redaction since no common y-coordinates were found.")
        doc = fitz.open(pdf_path)
        # # Step 3: Save the PDF file to the new path
        # doc.save(output_pdf_path)

        # # Close the document
        # doc.close()
        # print(f"Un Modified PDF saved to {output_pdf_path}")

        return doc

    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")

        for block in blocks:
            block_y_coord = block[1]

            # Check if the block's y-coordinate matches the common y-coordinates (allowing small tolerance)
            if any(abs(block_y_coord - common_y) < tolerance for common_y in common_y_coords):
                # Mark the block for redaction (removal)
                page.add_redact_annot(block[:4], fill=(1, 1, 1))  # Redact with white fill

        # Apply redactions for the current page
        page.apply_redactions()

    # # Save the modified PDF
    # doc.save(output_pdf_path)
    # doc.close()
    # print(f"Modified PDF saved to {output_pdf_path}")
    return doc


# Function to calculate the centroid of a text block
def getCentroid(bbox):
    y0, y1 = bbox[1], bbox[3]
    return (y0 + y1) / 2  # Centroid is the middle of y0 and y1

# Adjust the centroids of text spans based on adjacent text
def adjustCentroid(wordsData):
    adjustedData = wordsData.copy()
    for i in range(1, len(wordsData)):
        prevCentroid = wordsData[i - 1]['centroid']
        currCentroid = wordsData[i]['centroid']
        currY0, currY1 = wordsData[i]['y0'], wordsData[i]['y1']

        # If the current centroid falls between y0 and y1 of the previous text, adjust it
        if currY0 < prevCentroid < currY1:
            adjustedData[i]['centroid'] = prevCentroid

    return adjustedData

# Main function to extract words and apply centroid adjustments
def extract_text(pdfFilePath):
    # doc = fitz.open(pdfFilePath)
    doc = pdfFilePath
    outputData = []
    continuous_index = 0  # Initialize continuous index for the entire document

    # Extract text from the PDF at the word level
    for pageNum in range(doc.page_count):
        page = doc.load_page(pageNum)
        words = page.get_text("words")  # Extract words directly

        pageTextData = []
        for word_data in words:
            x0, y0, x1, y1, word, _, _, _ = word_data  # Extract coordinates and word text
            bbox = (x0, y0, x1, y1)

            # Calculate the centroid for each word
            centroid = getCentroid(bbox)

            # Append each word along with its details to pageTextData
            pageTextData.append({
                "text": word,
                "xCoord": x0,
                "yCoord": y0,
                "y0": y0,
                "y1": y1,
                "centroid": centroid,
                "bbox": bbox,
                "Page Number": pageNum + 1
            })

        # Adjust centroids based on adjacent text words
        pageTextData = adjustCentroid(pageTextData)

        # Sort the words by the adjusted centroid and X-coordinate
        pageTextData = sorted(pageTextData, key=lambda x: (x['centroid'], x['xCoord']))

        # Recalculate Start Index and End Index based on sorted order without resetting for each page
        for item in pageTextData:
            word_length = len(item["text"])
            item["Start Index"] = continuous_index
            item["End Index"] = continuous_index + word_length - 1
            continuous_index += word_length   # Increment by word length plus one for space

            # Append sorted and indexed word data to outputData
            outputData.append({
                "Page Number": item["Page Number"],
                "Text": item["text"],
                "Bounding Box": item["bbox"],
                "Start Index": item["Start Index"],
                "End Index": item["End Index"]
            })

    # Convert to DataFrame
    outputDf = pd.DataFrame(outputData)
    return outputDf

# Function to normalize sentences (remove leading/trailing spaces and normalize whitespace)
def normalize_sentence(sentence):
    return re.sub(r'\s+', ' ', sentence.strip())


# Function to find the index of a sentence in the full text
def find_index(sentence, full_text):
    try:
        return full_text.index(sentence)
    except ValueError:
        return -1  # Return -1 if the sentence is not found



# Initialize the Groq client
client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)

# Function to find added, deleted text, and explanation using Groq
def find_added_deleted_with_groq(old_text, new_text):

    # Convert to strings, handle NaN by converting to empty strings
    old_text = str(old_text) if not pd.isna(old_text) else ''
    new_text = str(new_text) if not pd.isna(new_text) else ''

    # Adjusted prompt for clearer response format
    prompt = (
        f"Given the following texts:\n"
        f"Old Text: '{old_text}'\n"
        f"New Text: '{new_text}'\n\n"
        f"Please identify the added and deleted text along with the summary and impact of the changes in terms of financial meaning on a scale of 1 to 10 where 1 being no change and 10 being major change in meaning in strictly the following JSON format:\n"
        f"{{\n"
        f"  'json_start': 'JSON Starts from here',\n"
        f"  'added_text': '...',\n"
        f"  'deleted_text': '...',\n"
        f"  'Change_summary': '...',\n"
        f"  'Impact': '...',\n"
        f"  'json_end': 'JSON Ends here'\n"
        f"}}"
    )


    retry_attempts = 3
    while retry_attempts > 0:
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:  # Catch general exceptions
            print(f"An error occurred: {e}")
            if 'rate_limit_exceeded' in str(e):
                # Try to extract retry time from the message
                retry_match = re.search(r'(\d+m\d+\.\d+s)', str(e))
                if retry_match:
                    retry_time = retry_match.group(1)
                    # Convert to seconds
                    minutes, seconds = map(float, re.findall(r'\d+\.\d+|\d+', retry_time))
                    retry_after = minutes * 60 + seconds
                else:
                    retry_after = 60  # Default to 60 seconds if we can't extract the retry time

                print(f"Rate limit reached. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                retry_attempts -= 1
            else:
                raise e  # If it's another error, raise it

    raise Exception("Exceeded retry limit. Please try again later.")


    # Extracting added, deleted text, and explanation from the model's response
    response_content = chat_completion.choices[0].message.content
    return response_content

def parse_response(response):

    return '', '', '','', response



def extract_first_number(text):
    # Use regex to find all numbers (including decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    # Return the first number as a string if available, otherwise return an empty string
    return numbers[0] if numbers else ''

def clean_text(text):
    if isinstance(text, str):
        # Remove any symbols before the first alphanumeric character and after the last alphanumeric character
        cleaned = re.sub(r'^[^\w]+', '', text)  # Remove leading non-alphanumeric characters
        cleaned = re.sub(r'[^\w]+$', '', cleaned)  # Remove trailing non-alphanumeric characters
        return cleaned
    else:
        return text  # If it's not a string, return the original value


# Function to extract text between two words
def extract_text_between(text, start_word, end_word):
    try:
        start_index = text.index(start_word) + len(start_word)
        end_index = text.index(end_word, start_index)
        return text[start_index:end_index].strip()
    except ValueError:
        return None  # Return None if the words are not found

# Function to split text into coherent sentences
def split_text_into_coherent_sentences(text):
    # Find all sequences of words and numbers that are separated by spaces or punctuation
    sentences = []
    current_sentence = []

    # Ensure the text is a string
    if not isinstance(text, str):
        return []

    # Split the text into words
    words = re.findall(r'\S+', text)  # This finds sequences of non-whitespace characters

    for word in words:
        # Check for specific patterns to identify sentence boundaries
        if word in ['.', ';', ':', ',']:  # If the word is punctuation, continue
            continue
        elif re.match(r'^\d', word):  # If it starts with a digit, we can consider it as part of the sentence
            current_sentence.append(word)
        elif word.startswith('('):  # Handle parenthesis
            continue
        else:
            current_sentence.append(word)

        # Add the current sentence if a new sentence should start
        if len(current_sentence) > 0 and (len(current_sentence) >= 5 or word.endswith('.')):  # Arbitrary sentence end logic
            sentences.append(' '.join(current_sentence).strip())
            current_sentence = []  # Reset for the next sentence

    # Add any remaining words as a final sentence
    if current_sentence:
        sentences.append(' '.join(current_sentence).strip())

    return sentences


def load_sentences_from_csv(input_csv):
    sentences = []
    with open(input_csv, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sentences.append(row['Sentence'])
    return sentences

def find_sentence_indexes(text, sentence, unwanted_chars_pattern):
    # Normalize the sentence for searching
    normalized_sentence = sentence#re.sub(unwanted_chars_pattern, '', sentence.strip()).strip().lower()
    indexes = []
    start = 0
    while True:
        index = text.lower().find(normalized_sentence, start)
        if index == -1:
            break
        indexes.append(str(index))
        start = index + 1  # Continue search after this index
    return indexes

def match_and_extract(df, cleaned_df, index_col, end_index_col, sentence_col, page_col, text_col, bounding_col):
    for i, row in cleaned_df.iterrows():
        # Calculate End Index
        extracted_sentence_length = len(row[sentence_col])  # Length of text in the extracted sentences
        End_Index = row[index_col] + extracted_sentence_length  # Calculate End Index

        # Find matches
        match = df[(df['Start Index'] <= row[index_col]) & (df['End Index'] >= row[index_col])]
        match_end = df[(df['Start Index'] <= End_Index) & (df['End Index'] >= End_Index)]

        # Check if both matches are the same
        if not match.empty and not match_end.empty and match.equals(match_end):
            cleaned_df.at[i, page_col] = match.iloc[0]['Page Number']
            cleaned_df.at[i, text_col] = match.iloc[0]['Text']
            cleaned_df.at[i, bounding_col] = match.iloc[0]['Bounding Box']
        else:
            # Extract all rows between match and match_end
            if not match.empty and not match_end.empty:
                start_index = match.index[0]
                end_index = match_end.index[0]
                if start_index > end_index:
                    start_index, end_index = end_index, start_index  # Ensure start_index is less than end_index

                extracted_rows = df.iloc[start_index:end_index + 1]

                # Ensure to handle NaN and float values
                cleaned_df.at[i, page_col] = '; '.join(extracted_rows['Page Number'].astype(str).tolist())
                cleaned_df.at[i, text_col] = '; '.join(extracted_rows['Text'].dropna().astype(str).tolist())
                cleaned_df.at[i, bounding_col] = '; '.join(extracted_rows['Bounding Box'].dropna().astype(str).tolist())
    return cleaned_df


def highlight_pdf(pdf_path, output_path, page_column, bbox_column, color, df):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # Get the page numbers and bounding boxes
        page_numbers = row[page_column].split(';') if pd.notna(row[page_column]) else []
        bboxes = row[bbox_column].split(';') if pd.notna(row[bbox_column]) else []

        # Iterate over each page number and corresponding bounding box
        for page_str, bbox_str in zip(page_numbers, bboxes):
            # Skip rows with missing or invalid data
            if not page_str.strip() or not bbox_str.strip():
                print(f"Skipping empty page or bbox at row {i}: page='{page_str}', bbox='{bbox_str}'")
                continue

            # Convert page number to integer
            try:
                page_number = int(page_str.strip())
            except ValueError:
                print(f"Invalid page number at row {i}: '{page_str}'")
                continue  # Skip if the page number is invalid

            # Convert bounding box from string to tuple of floats
            try:
                # Clean and parse the bounding box
                bbox_str_cleaned = bbox_str.strip().strip("()")  # Remove spaces and parentheses
                bbox_parts = [float(coord) for coord in bbox_str_cleaned.split(",")]
                if len(bbox_parts) != 4:
                    raise ValueError(f"Bounding box does not have 4 elements: {bbox_str_cleaned}")
                bbox = tuple(bbox_parts)  # Convert to tuple
            except (ValueError, TypeError) as e:
                print(f"Invalid bounding box format at row {i}: '{bbox_str}'. Error: {e}")
                continue  # Skip if the bbox format is incorrect

            # Get the specified page (zero-indexed)
            if 0 <= page_number - 1 < len(pdf_document):  # Check if the page number is within range
                page = pdf_document[page_number - 1]

                # Define the rectangle for the highlight
                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

                # Add a highlight to the page with the specified color
                highlight = page.add_rect_annot(rect)
                highlight.set_colors(stroke=color)  # Apply color
                highlight.update()
                # print(f"Highlighted bbox {bbox} on page {page_number} at row {i}")
            else:
                print(f"Page number {page_number} out of range for document with {len(pdf_document)} pages.")

    # Save the modified PDF
    pdf_document.save(output_path)
    pdf_document.close()
    print(f"Saved highlighted PDF to {output_path}")


def no_changes(pdf_file_path_new,pdf_file_path_old):
    modified_output_pdf_file_path_new = pdf_file_path_new
    modified_output_pdf_file_path_old = pdf_file_path_old

    # # Define the column names
    # columns = [
    #     'Sentence', 'New_Sentence_Index', 'Index', 'extracted sentences old',
    #     'extracted sentences new', 'Added Text', 'Deleted Text', 'Change_summary',
    #     'Impact', 'JSON Response', 'Old_Page_Number', 'Old_matched_text',
    #     'Old_Bounding Box', 'New_Page_Number', 'New_matched_text', 'New_Bounding Box'
    # ]

    # # Create an empty dataframe with the specified columns
    # df = pd.DataFrame(columns=columns)

    # # # Add a row with default values for 'Change_summary' and 'Impact'
    # # df = df.append({
    # #     'Change_summary': 'No Changes found',
    # #     'Impact': 0
    # # }, ignore_index=True)

    # # Create a new row with default values for 'Change_summary' and 'Impact'
    # new_row = pd.DataFrame([{
    #     'Change_summary': 'No changes found between the two PDF files',
    #     'Impact': 0
    # }], columns=columns)

    # # Concatenate the new row to the dataframe
    # df = pd.concat([df, new_row], ignore_index=True)

    # # Specify the file path
    # file_path = '/content/cleaned_filtered_differences_updated.csv'

    # # Save the dataframe to a CSV file
    # df.to_csv(file_path, index=False)

    # # print(f"CSV file created at {file_path}")

    # cleaned_filtered_differences_df_path = '/content/cleaned_filtered_differences_updated.csv'
    summary = 'No changes found between the two PDF files'
    impact =  '0'

    return modified_output_pdf_file_path_new, modified_output_pdf_file_path_old, summary, impact


def extract_and_save_sentence_indexes(text_old, text_new):
    # Assuming split_text_into_coherent_sentences is defined elsewhere and splits text into coherent sentences
    new_sentences = split_text_into_coherent_sentences(text_new)

    # Pattern to remove unwanted characters (if needed for processing)
    unwanted_chars_pattern = r'^\s*[\d\W]*|[\d\W]*\s*$'

    # Normalize new sentences only for searching
    new_sentences_normalized = [
        re.sub(unwanted_chars_pattern, '', s.strip()).strip().lower() for s in new_sentences
    ]

    # List to store sentence and its indices
    sentence_indexes = []

    # Find all occurrences of each normalized sentence in the original text_old and text_new
    for sentence in new_sentences_normalized:
        # Skip if sentence is empty, NaN, or doesn't contain valid alphanumeric characters
        if not sentence or not any(char.isalnum() for char in sentence) or (isinstance(sentence, float) and math.isnan(sentence)):
            continue

        # Find occurrences in the original text_old
        old_indexes = []
        start = 0
        while True:
            index = text_old.lower().find(sentence, start)  # Searching in original text_old
            if index == -1:
                break
            old_indexes.append(str(index))  # Append found index as string
            start = index + 1  # Continue search after this index

        # Find occurrences in the original text_new
        new_indexes = []
        start = 0
        while True:
            index = text_new.lower().find(sentence, start)  # Searching in original text_new
            if index == -1:
                break
            new_indexes.append(str(index))  # Append found index as string
            start = index + 1  # Continue search after this index

        # Save sentence, its old and new indexes if both have occurrences
        if old_indexes and new_indexes:
            sentence_indexes.append((sentence, ", ".join(old_indexes), ", ".join(new_indexes)))

    # # Write results to a CSV file
    # with open(output_csv, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Sentence", "Old_Indexes", "New_Indexes"])
    #     writer.writerows(sentence_indexes)

    # print(f"Sentence indexes saved to {output_csv}")
    return sentence_indexes

def process_csv(sentence_indexes):
    unique_rows = set()
    intermediate_rows = []
    deleted_rows = []

    # # Step 1: Remove duplicates and process indexes
    # with open(input_file, mode='r') as file:
    #     reader = csv.DictReader(file)

    #     for row in reader:
    #         row_tuple = (row['Sentence'], row['Old_Indexes'], row['New_Indexes'])
    #         if row_tuple not in unique_rows:
    #             unique_rows.add(row_tuple)
    #             intermediate_rows.append(row)
    #         else:
    #             deleted_rows.append(row)

    for row in sentence_indexes:
        row_tuple = (row[0], row[1], row[2])  # Unpack tuple into sentence, old indexes, new indexes
        if row_tuple not in unique_rows:
            unique_rows.add(row_tuple)
            intermediate_rows.append({'Sentence': row[0], 'Old_Indexes': row[1], 'New_Indexes': row[2]})
        else:
            deleted_rows.append({'Sentence': row[0], 'Old_Indexes': row[1], 'New_Indexes': row[2]})

    # print("Deleted rows:")
    # for deleted_row in deleted_rows:
    #     print(deleted_row)

    # Helper function to process columns and return individual index rows
    def process_column(rows, column_name):
        processed_rows = []
        for row in rows:
            sentence = row['Sentence']
            indexes = [index.strip() for index in row[column_name].split(",")]
            for index in indexes:
                processed_rows.append({'Sentence': sentence, 'Single_Index': int(index)})
        return processed_rows

    # Process both Old and New indexes
    old_index_rows = process_column(intermediate_rows, 'Old_Indexes')
    new_index_rows = process_column(intermediate_rows, 'New_Indexes')

    # Step 2: Sort rows by Single_Index
    old_index_rows.sort(key=lambda x: x['Single_Index'])
    new_index_rows.sort(key=lambda x: x['Single_Index'])

    # Step 3: Load old indexes into a dictionary
    old_indexes_dict = defaultdict(list)
    for row in old_index_rows:
        old_indexes_dict[row['Sentence']].append(row['Single_Index'])

    # Step 4: Create a list for the new rows with Old_indexes included
    rows_with_old_indexes = []
    for row in new_index_rows:
        old_indexes = old_indexes_dict.get(row['Sentence'], [])
        row['Old_indexes'] = ", ".join(map(str, old_indexes))
        rows_with_old_indexes.append(row)

    # Step 5: Drop duplicate sentences from the final rows
    sentence_counts = Counter(row['Sentence'] for row in rows_with_old_indexes)
    unique_final_rows = [row for row in rows_with_old_indexes if sentence_counts[row['Sentence']] == 1]

    # # Step 6: Write unique sentences to the final output file
    # with open(output_file_final, mode='w', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=list(unique_final_rows[0].keys()) if unique_final_rows else ['Sentence', 'Single_Index', 'Old_indexes'])
    #     writer.writeheader()
    #     writer.writerows(unique_final_rows)
    return pd.DataFrame(unique_final_rows)



def process_sentence_indexes(df):
    # with open(csv_file_path, mode='r') as file:
    #     reader = csv.reader(file)
    #     headers = next(reader)  # Skip header
    #     # Read sentence and split indexes for both 'Indexes' and 'New_Sentence_Index'
    #     data = [(row[0], row[1].split(", "), row[2].split(", ")) for row in reader]
    #Convert 'Indexes' and 'New_Sentence_Index' columns to lists of integers
    # data = [(row['Sentence'], row['Indexes'].split(", "), row['New_Sentence_Index'].split(", ")) for _, row in df.iterrows()]

    # data = [(row['Sentence'], row['New_Indexes'].split(", "), row['Old_indexes'].split(", ")) for _, row in df.iterrows()]


    # Ensure the necessary columns exist
    required_columns = ['Sentence', 'New_Indexes', 'Old_indexes']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Convert 'New_Indexes' and 'Old_indexes' columns to strings and then split them
    data = [(row['Sentence'], str(row['New_Indexes']).split(", "), str(row['Old_indexes']).split(", ")) for _, row in df.iterrows()]

    # Define a helper function to process index columns
    def process_column(data, column_index):
        start_index = 0
        last_index = -1  # Initialize last_index as -1 temporarily

        # Find the first row with a single index
        while start_index < len(data) and len(data[start_index][column_index]) != 1:
            start_index += 1

        # Check if a single-index row was found
        if start_index < len(data):
            last_index = int(data[start_index][column_index][0])  # Set last_index to the first single index found
        else:
            print("No single-index rows found to start processing.")
            return None

        # Process rows for the specified column
        result_column_data = []
        last_single_index_row = start_index  # Track last row with a single index
        delete_column = [""] * len(data)  # Initialize the Delete column with empty strings

        # Process the rows before start_index in reverse order for descending order
        for i in range(start_index - 1, -1, -1):
            sentence, indexes, new_indexes = data[i]
            indexes_to_process = indexes if column_index == 1 else new_indexes
            if len(indexes_to_process) > 1:
                # Convert available indexes to integers and sort them in descending order
                available_indexes = sorted((int(i) for i in indexes_to_process), reverse=True)
                # Choose the largest index that is less than the last_index
                chosen_index = next((index for index in available_indexes if index < last_index), None)
                if chosen_index is None:
                    print(f"No suitable index found for sentence: '{sentence}' to maintain descending order.")
                    return None
                result_column_data.append((sentence, chosen_index))
                last_index = chosen_index

        # Now process the rows starting from the first single-index row for ascending order
        for i in range(start_index, len(data)):
            sentence, indexes, new_indexes = data[i]
            indexes_to_process = indexes if column_index == 1 else new_indexes
            if len(indexes_to_process) == 1:
                index = int(indexes_to_process[0])
                # Check if the single index is greater than `last_index`
                if index < last_index:
                    # Mark this row with 'Delete' in delete_column and continue to the next row
                    delete_column[i] = "Delete"
                    print(f"Single index {index} is not greater than the previous index {last_index} for sentence: '{sentence}'. Marking as Delete.")
                    continue
                last_single_index_row = i  # Update the last single index row
                last_index = index  # Update `last_index` to the current single index
            else:
                # For multiple indexes, choose the smallest index greater than last_index
                available_indexes = sorted(int(i) for i in indexes_to_process if int(i) > last_index)
                if not available_indexes:
                    print(f"No suitable index found for sentence: '{sentence}' to maintain ascending order.")
                    # Mark rows from the last single index row up to the current row (not including last_single_index_row) with 'Delete'
                    for j in range(last_single_index_row + 1, i):
                        delete_column[j] = "Delete"
                    # Reset last_index to the single index of the last_single_index_row
                    last_index = int(data[last_single_index_row][column_index][0])
                    # Re-process current row with updated last_index to get a suitable index
                    available_indexes = sorted(int(i) for i in indexes_to_process if int(i) > last_index)
                    if not available_indexes:
                        delete_column[i] = "Delete"
                        continue

                # Select the suitable index for the current row and update last_index to it
                index = available_indexes[0]
                last_index = index  # Set last_index to the chosen index for the current row

            result_column_data.append((sentence, index))
            last_index = index

        return result_column_data, delete_column

    # Process each column individually
    processed_indexes, delete_indexes = process_column(data, column_index=1)
    processed_new_indexes, delete_new_indexes = process_column(data, column_index=2)

    # Check if both columns were successfully processed
    if processed_indexes is None or processed_new_indexes is None:
        print("Processing failed due to missing suitable indexes.")
        return

    # Combine results, filter out rows with 'Delete', and exclude the 'Delete' column
    result_data = []
    for i in range(len(processed_indexes)):
        sentence = processed_indexes[i][0]
        index = processed_indexes[i][1]
        new_index = processed_new_indexes[i][1]
        delete_flag = delete_indexes[i] if delete_indexes[i] == "Delete" else delete_new_indexes[i]

        # Only add rows that do not have 'Delete' in the delete_flag
        if delete_flag != "Delete":
            result_data.append((sentence, index, new_index))

    # # Save the filtered result data back to a CSV file, excluding the Delete column
    # with open(csv_file_path_new, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Sentence", "Index", "New_Sentence_Index"])  # Write header without 'Delete' column
    #     writer.writerows(result_data)  # Write processed data without 'Delete' rows

    # print(f"Processed data with ascending order saved to {csv_file_path_new}")


    # Return the processed DataFrame
    return pd.DataFrame(result_data, columns=["Sentence", "Index", "New_Sentence_Index"])



def extract_sentences_between_indexes(text_old_var, text_new_var, data1):
    # Pattern to remove unwanted characters
    unwanted_chars_pattern = r'^\s*[\d\W]*|[\d\W]*\s*$'

    # Normalize text_old and text_new

    text_old_normalized = text_old_var.lower()
    text_new_normalized = text_new_var.lower()

    # Load the CSV file
    # df = pd.read_csv(csv_file_path)
    df = data1

    # Filter the DataFrame to find valid sentences (non-empty)
    valid_sentences_df = df[df['Sentence'].notna() & (df['Sentence'] != '')]

    # Check if there are valid sentences
    if not valid_sentences_df.empty:
        # Convert 'Index' and 'New_Sentence_Index' columns to integers for easy slicing
        valid_sentences_df['Index'] = valid_sentences_df['Index'].astype(int)
        valid_sentences_df['New_Sentence_Index'] = valid_sentences_df['New_Sentence_Index'].astype(int)

        # Lists to store extracted sentences
        extracted_old_sentences = []
        extracted_new_sentences = []

        # Loop through pairs of consecutive rows for 'Index'
        for i in range(len(valid_sentences_df)):
            if i < len(valid_sentences_df) - 1:
                # Get the start and end indexes for the current row and next row
                start_index_old = valid_sentences_df['Index'].iloc[i]
                end_index_old = valid_sentences_df['Index'].iloc[i + 1]
                end_index_sentence_old = valid_sentences_df['Sentence'].iloc[i+1]

                start_index_new = valid_sentences_df['New_Sentence_Index'].iloc[i]
                end_index_new = valid_sentences_df['New_Sentence_Index'].iloc[i + 1]

                # Extract the text segments
                old_sentence_segment = text_old_normalized[start_index_old:end_index_old + len(end_index_sentence_old)]

                new_sentence_segment = text_new_normalized[start_index_new:end_index_new + len(end_index_sentence_old)]
            else:
                # For the last row, use the index of the last sentence to extract text till the end
                start_index_old = valid_sentences_df['Index'].iloc[i]
                last_sentence_old = valid_sentences_df['Sentence'].iloc[i]

                old_sentence_segment = text_old_normalized[start_index_old:start_index_old + len(last_sentence_old)]

                start_index_new = valid_sentences_df['New_Sentence_Index'].iloc[i]
                new_sentence_segment = text_new_normalized[start_index_new:start_index_new + len(last_sentence_old)]

            extracted_old_sentences.append(old_sentence_segment)
            extracted_new_sentences.append(new_sentence_segment)

        # Add the extracted sentences as new columns in the DataFrame
        valid_sentences_df['extracted sentences old'] = extracted_old_sentences
        valid_sentences_df['extracted sentences new'] = extracted_new_sentences

        # # Save the updated DataFrame with the extracted sentences to a new CSV file
        # valid_sentences_df.to_csv(output_csv_file_path, index=False)

        # # Print the path of the saved file
        # print(f"Extracted sentences saved to {output_csv_file_path}")
    else:
        print("No valid sentences found in the CSV file.")

    return valid_sentences_df



# def rename_column_in_csv(input_file, output_file, old_column_name, new_column_name):
#     # Read the original CSV file and rename the specified column
#     with open(input_file, mode='r') as infile:
#         reader = csv.DictReader(infile)
#         rows = list(reader)

#         # Rename the column in the fieldnames
#         fieldnames = [new_column_name if col == old_column_name else col for col in reader.fieldnames]

#     # Write to a new CSV file with the updated column name
#     with open(output_file, mode='w', newline='') as outfile:
#         writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in rows:
#             # Create a new row with the updated column name
#             new_row = {new_column_name if col == old_column_name else col: value for col, value in row.items()}
#             writer.writerow(new_row)
# import csv

# def rename_column_in_csv(data, old_column_name, new_column_name):
#     """
#     Renames a specified column in a list of dictionaries.

#     Parameters:
#         data (list): List of dictionaries representing rows in a CSV format.
#         old_column_name (str): The name of the column to rename.
#         new_column_name (str): The new name for the column.

#     Returns:
#         list: Updated list of dictionaries with the renamed column.
#     """
#     # Rename the column in the headers
#     fieldnames = [new_column_name if col == old_column_name else col for col in data[0].keys()]

#     # Update each row in the data
#     updated_data = []
#     for row in data:
#         # Create a new row with the updated column name
#         new_row = {new_column_name if col == old_column_name else col: value for col, value in row.items()}
#         updated_data.append(new_row)

#     return updated_data
def rename_column_in_csv(data, old_column_name, new_column_name):
    """
    Renames a specified column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        old_column_name (str): The name of the column to rename.
        new_column_name (str): The new name for the column.

    Returns:
        pd.DataFrame: Updated DataFrame with the renamed column.
    """
    # Use the rename method to rename the column
    data = data.rename(columns={old_column_name: new_column_name})

    return data


def longest_increasing_subsequence_prioritize_x(x_seq, y_seq):
    n = len(y_seq)
    if n == 0:
        return [], []

    # DP arrays to store the longest increasing subsequence ending at each index
    dp = [1] * n
    prev = [-1] * n  # Store previous index in sequence for reconstruction
    max_length, max_end_index = 1, 0

    # Build DP table for LIS on y-values with increasing x-values
    for i in range(1, n):
        for j in range(i):
            if y_seq[i] > y_seq[j] and x_seq[i] > x_seq[j] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
                prev[i] = j
        if dp[i] > max_length:
            max_length = dp[i]
            max_end_index = i

    # Reconstruct the longest increasing subsequence
    lis_x, lis_y = [], []
    while max_end_index != -1:
        lis_x.append(x_seq[max_end_index])
        lis_y.append(y_seq[max_end_index])
        max_end_index = prev[max_end_index]
    lis_x.reverse()
    lis_y.reverse()
    return lis_x, lis_y

# Function to convert comma-separated float values to integers
def convert_to_ints(value):
    # Ensure the value is a string, then process
    if not isinstance(value, str):
        value = str(value)
    # Split by comma, convert each to int, and join back as comma-separated string
    return ', '.join(str(int(float(x.strip()))) for x in value.split(','))



def update_extracted_sentences(df_with_spacing_var, df_without_spacing_var, output_csv):
    # Load the source and target CSV files
    source_df = df_with_spacing_var
    target_df = df_without_spacing_var

    # Print the shapes of the dataframes for debugging
    print(f"Source DataFrame shape: {source_df.shape}")
    print(f"Target DataFrame shape: {target_df.shape}")

    # Merge the two DataFrames on the 'Sentence' column
    merged_df = target_df.merge(
        source_df[['Sentence', 'extracted sentences old', 'extracted sentences new']],
        on='Sentence',
        how='left',
        suffixes=('', '_source')
    )

    # Update the 'extracted sentences old' and 'extracted sentences new' in target DataFrame
    merged_df['extracted sentences old'] = merged_df['extracted sentences old_source'].combine_first(merged_df['extracted sentences old'])
    merged_df['extracted sentences new'] = merged_df['extracted sentences new_source'].combine_first(merged_df['extracted sentences new'])

    # Drop the temporary columns used for merging
    merged_df.drop(columns=['extracted sentences old_source', 'extracted sentences new_source'], inplace=True)

    # Save the updated DataFrame to a new CSV file
    merged_df.to_csv(output_csv, index=False)

    print(f"Updated data saved to {output_csv}")
    return merged_df


##below is original code for span level
def add_space_to_df(df):
    # Add a space at the end of each text in the 'Text' column
    df['Text'] = df['Text'] + ' '
    cumulative_offset = 2  # Initialize the cumulative offset
    # # Increment the 'End Index' of the first row and propagate the change to maintain continuity
    df.loc[0, 'End Index'] += 1  # Increment the first row's 'End Index' by 1

    # Adjust the 'Start Index' and 'End Index' for subsequent rows to maintain continuity
    for i in range(1, len(df)):
        df.loc[i, 'Start Index'] = df.loc[i - 1, 'End Index']  # Set 'Start Index' to be continuous
        df.loc[i, 'End Index'] = df.loc[i, 'End Index'] + cumulative_offset        # Adjust 'End Index' accordingly
        cumulative_offset += 1
    return df

# def main(pdf_file_path_new , pdf_file_path_old):
def main(pdf_file_path_new , pdf_file_path_old, modified_output_pdf_file_path_new, modified_output_pdf_file_path_old):
    print("Comparing pdf files")


    # Main execution
    pdf_path = pdf_file_path_new
    # output_pdf_path_new = '/content/new_modified.pdf'

    # Step 1: Extract common y-coordinates (headers/footers)
    common_y_coords = extract_common_y_coords(pdf_path)

    # Step 2: Remove the text at those y-coordinates
    output_pdf_path_new_without_headers = remove_header_footer(pdf_path, common_y_coords)


    # Main execution
    pdf_path = pdf_file_path_old
    # output_pdf_path_old = '/content/old_modified.pdf'

    # Step 1: Extract common y-coordinates (headers/footers)
    common_y_coords = extract_common_y_coords(pdf_path)

    # Step 2: Remove the text at those y-coordinates
    output_pdf_path_old_without_headers = remove_header_footer(pdf_path, common_y_coords)

    new_df = extract_text(output_pdf_path_new_without_headers)
    old_df = extract_text(output_pdf_path_old_without_headers)

    new_df = add_space_to_df(new_df)
    old_df = add_space_to_df(old_df)
    # # Specify the output file path
    # output_file_path = 'new_df.csv'
    # # Save the DataFrame to a CSV file
    # new_df.to_csv(output_file_path, index=False)


    # # Specify the output file path
    # output_file_path = 'old_df.csv'
    # # Save the DataFrame to a CSV file
    # old_df.to_csv(output_file_path, index=False)

    # # Load the input CSV file
    # input_path = '/content/new_df.csv'
    # output_path = '/content/processed_text.csv'
    # df = pd.read_csv(input_path)

    # # Step 1: Concatenate all 'Text' into a single string, converting each item to a string and handling NaN values
    # concatenated_text = ''.join(new_df['Text'])#''.join(df['Text'].fillna('').astype(str).tolist())

    # # Step 2: Split the concatenated text into rows based on 'Start Index' and 'End Index'
    # # We'll create a new DataFrame for this
    # split_texts = []
    # for _, row in df.iterrows():
    #     start = int(row['Start Index'])
    #     end = int(row['End Index'])
    #     split_text = concatenated_text[start:end]  # Extract the substring
    #     split_texts.append({
    #         'Page Number': row['Page Number'],
    #         'Start Index': start,
    #         'End Index': end,
    #         'Text': split_text,
    #         'Bounding Box': row['Bounding Box']
    #     })

    # # Convert list of dictionaries to DataFrame
    # processed_df = pd.DataFrame(split_texts)

    # # Step 3: Save the result to a new CSV file
    # processed_df.to_csv(output_path, index=False)
    # print(f"Processed data saved to {output_path}")

    # Concatenate all text into a single string
    text_new = ''.join(new_df['Text'])
    # print('text_new',text_new)
    text_old = ''.join(old_df['Text'])
    # print('text_old',text_old)


    # # Create a DataFrame
    # df = pd.DataFrame({'text_new': [text_new]})

    # # Save to CSV
    # df.to_csv('/content/text_new.csv', index=False)

    # extract_and_save_sentence_indexes(text_old, text_new, output_csv ="sentence_indexes.csv")
    sentence_indexes = extract_and_save_sentence_indexes(text_old, text_new)


    # # Specify the input and output file paths
    # input_file = '/content/sentence_indexes.csv'
    # output_file_final = '/content/sentence_indexes_unique_sentences.csv'

    # # Call the function to process the file
    # process_csv(input_file, output_file_final)
    sentence_indexes_unique_sentences = process_csv(sentence_indexes)


    # # Specify the input and output file paths
    # input_file = '/content/sentence_indexes_unique_sentences.csv'
    # output_file = '/content/sentence_indexes_unique_sentences_renamed.csv'  # Output file with updated column name

    # # Call the function to rename the column
    # rename_column_in_csv(input_file, output_file, 'Single_Index', 'New_Indexes')

    sentence_indexes_unique_sentences_renamed = rename_column_in_csv(sentence_indexes_unique_sentences ,'Single_Index', 'New_Indexes')


    # # Load the data
    # file_path = '/content/sentence_indexes_unique_sentences_renamed.csv'

    # Load the data
    # file_path = '/content/sentence_indexes_unique_sentences_with_spacing_renamed.csv'
    # data = pd.read_csv(file_path)
    data = sentence_indexes_unique_sentences_renamed

    # Extract x (row index) and y (Old_indexes values) for processing
    x_values, y_values, annotations = [], [], []
    for index, row in data.iterrows():
        old_indexes = [int(i.strip()) for i in str(row['Old_indexes']).split(',')] if pd.notna(row['Old_indexes']) else []
        x_values.extend([index] * len(old_indexes))  # x is the row index
        y_values.extend(old_indexes)                 # y is the index value
        annotations.extend([(index, old_index) for old_index in old_indexes])  # annotations for plotting

    # Calculate IQR and filter outliers in y_values
    y_array = np.array(y_values)
    q1, q3 = np.percentile(y_array, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 0.5 * iqr
    upper_bound = q3 + 0.5 * iqr

    filtered_x_values = [x for x, y in zip(x_values, y_values) if lower_bound <= y <= upper_bound]
    filtered_y_values = [y for y in y_values if lower_bound <= y <= upper_bound]

    # Function to find the longest increasing subsequence in y_values with priority on longest x_values

    # Get the longest increasing subsequence prioritizing x-values
    longest_x_sequence, longest_y_sequence = longest_increasing_subsequence_prioritize_x(filtered_x_values, filtered_y_values)

    data['Longest_Ascending_Sequence'] = ""
    for x, y in zip(longest_x_sequence, longest_y_sequence):
        if data.loc[x, 'Longest_Ascending_Sequence']:
            data.loc[x, 'Longest_Ascending_Sequence'] += f", {y}"
        else:
            data.loc[x, 'Longest_Ascending_Sequence'] = str(y)

    # Sort each cell's values and remove duplicates
    data['Longest_Ascending_Sequence'] = data['Longest_Ascending_Sequence'].apply(
        lambda x: ', '.join(sorted(set(x.split(', ')), key=int)) if x else ''
    )

    df = data

    # Drop the original 'Old_indexes' column and rename the new column
    df.drop(columns=['Old_indexes'], inplace=True)
    df.rename(columns={'Longest_Ascending_Sequence': 'Old_indexes'}, inplace=True)

    # Drop rows where 'Old_indexes' is empty
    df = df[df['Old_indexes'] != '']
    # Drop rows where 'Old_indexes' (formerly 'Longest_Ascending_Sequence') is blank
    df.dropna(subset=['Old_indexes'], inplace=True)

    # Save the final modified DataFrame to a new CSV file
    # output_file_path = '/content/sentence_indexes_unique_sentences_renamed_sorted.csv'
    # df.to_csv(output_file_path, index=False)

    # Define file paths
    # file1 = '/content/sentence_indexes_unique_sentences_renamed_sorted.csv'

    # Load, convert, and save the first file
    # # df1 = pd.read_csv(file1)
    # df1 = df
    # df1['Old_indexes'] = df1['Old_indexes'].apply(convert_to_ints)

    # df1.to_csv(file1, index=False)

    df['Old_indexes'] = df['Old_indexes'].apply(convert_to_ints)




    df.to_csv('/content/sentence_indexes_unique_sentences_renamed_sorted.csv', index=False)
    # process_sentence_indexes('/content/sentence_indexes_unique_sentences_renamed_sorted.csv', '/content/sentence_indexes_new.csv')
    sentence_indexes_new = process_sentence_indexes(df)


    # # Load the CSV file
    # file_path = '/content/sentence_indexes_new.csv'
    # df = pd.read_csv(file_path)
    df2 = sentence_indexes_new

    # Rename the columns
    df2.rename(columns={'Index': 'New_Sentence_Index', 'New_Sentence_Index': 'Index'}, inplace=True)

    # # Save the modified DataFrame back to the CSV file
    # df.to_csv(file_path, index=False)

    # csv_file_path = '/content/sentence_indexes_new.csv'
    # output_csv_file_path = '/content/sentence_indexes_new_extracted.csv'
    # df_without_spacing = extract_sentences_between_indexes(text_old, text_new, csv_file_path, output_csv_file_path)
    # df = extract_sentences_between_indexes(text_old, text_new, csv_file_path, output_csv_file_path)
    df = extract_sentences_between_indexes(text_old, text_new, df2)

    df_cleaned = df[
        (df['extracted sentences old'].notna()) &
        (df['extracted sentences old'] != '') &
        (df['extracted sentences new'].notna()) &
        (df['extracted sentences new'] != '') &
        (df['extracted sentences old'] != df['extracted sentences new'])
    ]

    if df_cleaned.empty:
        print("No changes between the two PDF files.")
        # modified_output_pdf_file_path_new, modified_output_pdf_file_path_old, cleaned_filtered_differences_df_path = no_changes(pdf_file_path_new, pdf_file_path_old)
        modified_output_pdf_file_path_new, modified_output_pdf_file_path_old, summary, impact_level = no_changes(pdf_file_path_new, pdf_file_path_old)
    else:
        df = df_cleaned

        # Apply the function to each row of the DataFrame
        added_deleted_results = df.apply(
          lambda row: find_added_deleted_with_groq(row['extracted sentences old'], row['extracted sentences new']),
          # lambda row: find_added_deleted_with_groq(row['Old Start Heading'], row['New Start Heading']),
          axis=1
        )

        # Create new columns in the DataFrame
        df['Added Text'], df['Deleted Text'],  df['Change_summary'], df['Impact'],df['JSON Response'] = zip(*added_deleted_results.apply(parse_response))


        # Extract texts and create new columns
        df['Added Text'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'added_text', 'deleted_text'))
        df['Deleted Text'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'deleted_text', 'Change_summary'))
        df['Change_summary'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'Change_summary', 'Impact'))
        df['Impact'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'Impact', 'json_end'))

        df['Impact'] = df['Impact'].apply(lambda x: extract_first_number(str(x)))

        # Drop rows where 'Impact' contains '1', '1.0', '0', or '0.0' (including variations)
        df_filtered = df[~df['Impact'].astype(str).str.contains(r'^\s*(1|1\.0|0|0\.0)\s*(\(.*\))?$', na=False, case=False)]

        df = df_filtered.reset_index(drop=True)

        # Apply the cleaning function to the specified columns
        columns_to_clean = ['Added Text', 'Deleted Text', 'Change_summary']
        for column in columns_to_clean:
          df[column] = df[column].apply(clean_text)

        # df.to_csv('/content/cleaned_filtered_differences.csv', index=False)

        # Load the files
        # cleaned_df = pd.read_csv('/content/cleaned_filtered_differences.csv')
        cleaned_df = df
        # old_df = pd.read_csv('/content/old_df.csv')
        # new_df = pd.read_csv('/content/new_df.csv')

        # Initialize new columns for results
        cleaned_df['Old_Page_Number'] = None
        cleaned_df['Old_matched_text'] = None
        cleaned_df['Old_Bounding Box'] = None
        cleaned_df['New_Page_Number'] = None
        cleaned_df['New_matched_text'] = None
        cleaned_df['New_Bounding Box'] = None

        # Match and extract for old_df
        cleaned_df = match_and_extract(old_df, cleaned_df, 'Index', 'End Index1', 'extracted sentences old',
                          'Old_Page_Number', 'Old_matched_text', 'Old_Bounding Box')

        # Match and extract for new_df
        cleaned_df = match_and_extract(new_df, cleaned_df, 'New_Sentence_Index', 'End_Index_new1', 'extracted sentences new',
                          'New_Page_Number', 'New_matched_text', 'New_Bounding Box')

        # # Save the updated dataframe to CSV
        # cleaned_df.to_csv('/content/cleaned_filtered_differences_cleaned.csv', index=False)


        # # Load the CSV file
        # file_path = '/content/cleaned_filtered_differences_cleaned.csv'
        # df = pd.read_csv(file_path)
        df = cleaned_df

        # Define conditions for 'Added Text' and 'Deleted Text'
        added_text_condition = (df['Added Text'].isna()) | (df['Added Text'] == 'none') | (df['Added Text'] == 'No added text')
        deleted_text_condition = (df['Deleted Text'].isna()) | (df['Deleted Text'] == 'none') | (df['Deleted Text'] == 'No deleted text')

        # Drop rows where both conditions are satisfied
        df = df[~(added_text_condition & deleted_text_condition)]

        # Save the updated DataFrame back to a CSV file
        # df.to_csv('/content/cleaned_filtered_differences_updated.csv', index=False)

        # cleaned_filtered_differences_df_path = '/content/cleaned_filtered_differences_updated.csv'

        if df.empty:
            print("No changes between the two PDF files.")
            modified_output_pdf_file_path_new, modified_output_pdf_file_path_old, summary, impact_level = no_changes(pdf_file_path_new, pdf_file_path_old)
        else:

            modified_output_pdf_file_path_new = modified_output_pdf_file_path_new
            modified_output_pdf_file_path_old = modified_output_pdf_file_path_old

            # Load the CSV file
            # df = pd.read_csv(cleaned_filtered_differences_df_path)
            df = cleaned_df





            # Call the highlight_pdf function for the old PDF
            highlight_pdf(pdf_file_path_old, modified_output_pdf_file_path_old,
                          page_column='Old_Page_Number',
                          bbox_column='Old_Bounding Box',
                          color=(1, 0, 0),  # Red color for old PDF
                          df=df)

            # Call the highlight_pdf function for the new PDF
            highlight_pdf(pdf_file_path_new, modified_output_pdf_file_path_new,
                          page_column='New_Page_Number',
                          bbox_column='New_Bounding Box',
                          color=(0, 1, 0),  # Green color for new PDF
                          df=df)

            df.dropna(inplace=True)
            summary = df['Change_summary'].tolist()
            try:
                impact_score_list = df['Impact'].tolist()
                print(impact_score_list)
                max_score = max(impact_score_list)
                max_score = int(max_score)
                if max_score<4:
                    impact_level='Low'
                elif max_score<7:
                    impact_level = 'Medium'
                elif max_score<=10:
                    impact_level='High'
            except:
                impact_level = 'Low'

    return modified_output_pdf_file_path_new, modified_output_pdf_file_path_old, summary, impact_level



