import fitz  # PyMuPDF
from collections import Counter
import csv
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
os.environ['GROQ_API_KEY'] = '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' # Set the GROQ_API_KEY directly in the environment


from groq import Groq
import json
import re


import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)


def extract_headings_from_pdf(pdf_file):
    # Open the PDF file
    doc = fitz.open(pdf_file)

    # Collect all font sizes and text with their positions
    font_sizes = []
    text_positions = {}
    extracted_headings = []
    header_footer_extracted = []

    # Loop through the pages of the PDF to gather font sizes and positions
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") == 0:  # Only process text blocks
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])  # Collect font sizes
                        text = span["text"].strip()
                        bbox = span["bbox"]  # Get bounding box for the text
                        position = (bbox[0], bbox[1])  # Use (x0, y0) for position

                        # Store text by position
                        if position not in text_positions:
                            text_positions[position] = []
                        text_positions[position].append({
                            "text": text,
                            "bbox": bbox,
                            "size": span["size"]
                        })

    # Count occurrences of each font size
    font_size_counts = Counter(font_sizes)

    # Identify the font size with the maximum occurrences (likely paragraph text)
    most_common_size, most_common_count = font_size_counts.most_common(1)[0] if font_size_counts else (None, 0)

    # Get unique font sizes and sort them
    unique_font_sizes = sorted(set(font_sizes))
    print (unique_font_sizes)

    # Determine number of levels based on unique font sizes
    num_levels = len(unique_font_sizes)

    # Create a mapping of font sizes to heading levels, excluding the most common size
    size_to_level = {}
    for level, size in enumerate(unique_font_sizes):
        if size != most_common_size:  # Exclude the most common size
            size_to_level[size] = level + 1  # Level starts at 1

    # Identify header/footer text by checking for repetition across positions
    header_footer_texts = set()
    header_footer_y_coords = set()  # To track Y-coordinates of headers/footers
    for texts in text_positions.values():
        all_texts = [entry['text'] for entry in texts]
        common_texts = Counter(all_texts)
        for text, count in common_texts.items():
            if count > 1:  # If this text appears on multiple pages (likely header/footer)
                header_footer_texts.add(text)
                # Also store the Y-coordinate of this text
                for entry in texts:
                    if entry["text"] == text:
                        header_footer_y_coords.add(entry["bbox"][1])  # Add Y-coordinate

    # Reset for another pass to extract headings with thresholds
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") == 0:  # Only process text blocks
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_size = span["size"]
                        text = span["text"].strip()
                        bbox = span["bbox"]  # Get bounding box for the text
                        position = (bbox[0], bbox[1])  # Use (x0, y0) for position
                        y_coord = bbox[1]  # Y-coordinate for the current text

                        # Calculate the centroid (X, Y) of the bounding box
                        centroid_x = (bbox[0] + bbox[2]) / 2
                        centroid_y = (bbox[1] + bbox[3]) / 2

                        # Determine the heading level based on the size_to_level mapping
                        level = size_to_level.get(font_size, None)  # Default to None if not a heading

                        # Check if this is header/footer or text with the same Y-coordinate
                        if text in header_footer_texts or y_coord in header_footer_y_coords:
                            header_footer_extracted.append({
                                "Page Number": page_num + 1,
                                "Text": text,
                                "Y-Coordinate": y_coord,
                                "Centroid": (centroid_x, centroid_y),
                                "Level": level or "Body"
                            })
                        else:
                            # Add to main extracted headings if it's not a header/footer
                            if level is not None and text and y_coord not in header_footer_y_coords:
                                extracted_headings.append({
                                    "Page Number": page_num + 1,
                                    "Level": level,
                                    "Text": text,
                                    "Y-Coordinate": y_coord,
                                    "Centroid": (centroid_x, centroid_y)  # Store the centroid
                                })

    # Sort the extracted headings by Y-coordinate for each page
    sorted_headings = {}
    for heading in extracted_headings:
        page_num = heading['Page Number']
        if page_num not in sorted_headings:
            sorted_headings[page_num] = []
        sorted_headings[page_num].append(heading)

    # Sort headings on each page by Y-coordinate
    for page_num in sorted_headings:
        sorted_headings[page_num].sort(key=lambda h: h['Y-Coordinate'])

    return sorted_headings, header_footer_extracted, num_levels



def save_headings_to_df(headings_by_page):
    # Prepare the data for the DataFrame
    data = []

    for page_num, headings in headings_by_page.items():
        for heading in headings:
            data.append({
                'Page Number': heading['Page Number'],
                'Level': heading['Level'],
                'Text': heading['Text'],
                'Y-Coordinate': heading['Y-Coordinate'],
                'Centroid X': heading['Centroid'][0],  # Centroid X
                'Centroid Y': heading['Centroid'][1]   # Centroid Y
            })

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data, columns=['Page Number', 'Level', 'Text', 'Y-Coordinate', 'Centroid X', 'Centroid Y'])

    return df

def save_header_footer_to_df(header_footer_texts):
    # Prepare the data for the DataFrame
    data = []

    for entry in header_footer_texts:
        data.append({
            'Page Number': entry['Page Number'],
            'Text': entry['Text'],
            'Y-Coordinate': entry['Y-Coordinate'],
            'Centroid X': entry['Centroid'][0],  # Centroid X
            'Centroid Y': entry['Centroid'][1],  # Centroid Y
            'Level': entry['Level']
        })

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data, columns=['Page Number', 'Text', 'Y-Coordinate', 'Centroid X', 'Centroid Y', 'Level'])

    return df

#################

def extract_text_between_headings(headingsDf, headerFooterDf, pdfFilePath):

    doc = fitz.open(pdfFilePath)

    # Function to check if text is in header/footer by comparing the text and Centroid Y
    def isInHeaderFooter(text, centroid):
        return ((headerFooterDf['Text'] == text) & (headerFooterDf['Centroid Y'] == centroid)).any()

    # Function to calculate the centroid of a text block
    def getCentroid(bbox):
        y0, y1 = bbox[1], bbox[3]
        return (y0 + y1) / 2  # Centroid is the middle of y0 and y1

    # Adjust the centroids of text spans based on adjacent text
    def adjustCentroid(centroids):
        adjustedCentroids = centroids.copy()
        for i in range(1, len(centroids)):
            prevCentroid = centroids[i - 1]['centroid']
            currCentroid = centroids[i]['centroid']
            currY0, currY1 = centroids[i]['y0'], centroids[i]['y1']

            # If the current centroid falls between y0 and y1 of the previous text, adjust it
            if currY0 < prevCentroid < currY1:
                adjustedCentroids[i]['centroid'] = prevCentroid

        return adjustedCentroids



    # Extract all text from the PDF and organize it by page in the order as it appears
    pdfTextData = {}
    for pageNum in range(doc.page_count):
        page = doc.load_page(pageNum)
        blocks = page.get_text("dict")["blocks"]

        pageTextData = []
        for block in blocks:
            if block.get("type") == 0:  # Only process text blocks (not images, lines, etc.)
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        bbox = span["bbox"]
                        xCoord = bbox[0]  # X-coordinate of the text block
                        yCoord = bbox[1]  # Top Y-coordinate
                        centroid = getCentroid(bbox)  # Calculate centroid

                        if not isInHeaderFooter(text, centroid):
                            pageTextData.append({
                                "text": text,
                                "xCoord": xCoord,
                                "yCoord": yCoord,
                                "y0": bbox[1],  # Top Y-coordinate of the span
                                "y1": bbox[3],  # Bottom Y-coordinate of the span
                                "centroid": centroid
                            })

        # Adjust centroids based on adjacent text spans
        pageTextData = adjustCentroid(pageTextData)

        # Sort the text on the page by the adjusted centroid and X-coordinate
        pageTextData = sorted(pageTextData, key=lambda x: (x['centroid'], x['xCoord']))
        pdfTextData[pageNum + 1] = pageTextData  # Store text for each page in order

    # Function to get text between two Centroid Y on a page
    def getTextBetweenY(pageData, yStart, yEnd):
        return " ".join([item['text'] for item in pageData if yStart <= item['centroid'] < yEnd])

    # Group text between headings and across pages, maintaining order
    extractedTexts = []

    for i in range(len(headingsDf) - 1):
        startHeading = headingsDf.iloc[i]
        endHeading = headingsDf.iloc[i + 1]

        startPage = startHeading['Page Number']
        startY = startHeading['Centroid Y']

        endPage = endHeading['Page Number']
        endY = endHeading['Centroid Y']

        textBetweenHeadings = ""

        # Case when the headings are on the same page
        if startPage == endPage:
            pageData = pdfTextData[startPage]
            textBetweenHeadings = getTextBetweenY(pageData, startY, endY)

        # Case when the headings span multiple pages
        else:
            # Get text from the start page (from startY to the bottom of the page)
            pageData = pdfTextData[startPage]
            textBetweenHeadings += getTextBetweenY(pageData, startY, float('inf'))

            # Get text from all pages in between startPage and endPage
            for pageNum in range(startPage + 1, endPage):
                pageData = pdfTextData[pageNum]
                textBetweenHeadings += " " + getTextBetweenY(pageData, -float('inf'), float('inf'))

            # Get text from the end page (from the top of the page to endY)
            pageData = pdfTextData[endPage]
            textBetweenHeadings += " " + getTextBetweenY(pageData, -float('inf'), endY)

        # Store the text between the two headings
        extractedTexts.append({
            "Start Heading": startHeading['Text'],
            "End Heading": endHeading['Text'],
            "Text Between Headings": textBetweenHeadings.strip()
        })

    # Handle the last heading in the PDF (no subsequent heading)
    lastHeading = headingsDf.iloc[-1]
    lastPage = lastHeading['Page Number']
    lastY = lastHeading['Centroid Y']

    lastText = ""

    # Get text from the last heading page until the end of that page
    pageData = pdfTextData[lastPage]
    lastText += getTextBetweenY(pageData, lastY, float('inf'))

    # Get text from all subsequent pages until the end of the document
    for pageNum in range(lastPage + 1, doc.page_count + 1):
        pageData = pdfTextData[pageNum]
        lastText += " " + getTextBetweenY(pageData, -float('inf'), float('inf'))

    # Append the text for the last heading
    extractedTexts.append({
        "Start Heading": lastHeading['Text'],
        "End Heading": "End of Document",  # Since there's no subsequent heading
        "Text Between Headings": lastText.strip()
    })


    text_between_headings_in_order1 = pd.DataFrame(extractedTexts)

    return text_between_headings_in_order1


######################

# Function to clear matching text only
def clear_matching_texts(df):
    for i in range(len(df)):
        current_text = str(df.at[i, 'Text Between Headings']) if pd.notna(df.at[i, 'Text Between Headings']) else ''
        current_start_heading = str(df.at[i, 'Start Heading']) if pd.notna(df.at[i, 'Start Heading']) else ''
        current_end_heading = str(df.at[i, 'End Heading']) if pd.notna(df.at[i, 'End Heading']) else ''

        # Check previous row if exists
        if i > 0:
            prev_start_heading = str(df.at[i - 1, 'Start Heading']) if pd.notna(df.at[i - 1, 'Start Heading']) else ''
            prev_end_heading = str(df.at[i - 1, 'End Heading']) if pd.notna(df.at[i - 1, 'End Heading']) else ''
            prev_combined = prev_start_heading + ' ' + prev_end_heading

            # Check if current text matches with previous headings
            if (prev_start_heading in current_text):
                current_text = current_text.replace(prev_start_heading, '')
            if (prev_end_heading in current_text):
                current_text = current_text.replace(prev_end_heading, '')
            if (prev_combined in current_text):
                current_text = current_text.replace(prev_combined, '')

        # Check next row if exists
        if i < len(df) - 1:
            next_start_heading = str(df.at[i + 1, 'Start Heading']) if pd.notna(df.at[i + 1, 'Start Heading']) else ''
            next_end_heading = str(df.at[i + 1, 'End Heading']) if pd.notna(df.at[i + 1, 'End Heading']) else ''
            next_combined = next_start_heading + ' ' + next_end_heading

            # Check if current text matches with next headings
            if (next_start_heading in current_text):
                current_text = current_text.replace(next_start_heading, '')
            if (next_end_heading in current_text):
                current_text = current_text.replace(next_end_heading, '')
            if (next_combined in current_text):
                current_text = current_text.replace(next_combined, '')

        # Check current row headings
        combined_current = current_start_heading + ' ' + current_end_heading

        if (current_start_heading in current_text):
            current_text = current_text.replace(current_start_heading, '')
        if (current_end_heading in current_text):
            current_text = current_text.replace(current_end_heading, '')
        if (combined_current in current_text):
            current_text = current_text.replace(combined_current, '')

        # Update the DataFrame with the modified text
        df.at[i, 'Text Between Headings'] = current_text.strip()  # Remove leading/trailing whitespace
        return df

####################


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
        f"Please identify the added and deleted text along with the impact of the changes in meaning on a scale of 1 to 10 where 1 being no change and 10 being major change in meaning in strictly the following JSON format:\n"
        f"{{\n"
        f"  'json_start': 'JSON Starts from here',\n"
        f"  'added_text': '...',\n"
        f"  'deleted_text': '...',\n"
        f"  'Change_summary': '...',\n"
        f"  'Impact': '...',\n"
        f"  'json_end': 'JSON Ends here'\n"
        f"}}"
    )




    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )

    # Extracting added, deleted text, and explanation from the model's response
    response_content = chat_completion.choices[0].message.content
    return response_content


def parse_response(response):
    return '', '', '','', response

# Function to extract text between two words
def extract_text_between(text, start_word, end_word):
    try:
        start_index = text.index(start_word) + len(start_word)
        end_index = text.index(end_word, start_index)
        return text[start_index:end_index].strip()
    except ValueError:
        return None  # Return None if the words are not found


##################################
# Function to extract only numbers from a string
def extract_numbers(text):
    # Use regex to find all numbers (including decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    # Join the numbers into a single string, separated by spaces
    return ' '.join(numbers)

###################


# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        # Remove any symbols before the first alphanumeric character and after the last alphanumeric character
        cleaned = re.sub(r'^[^\w]+', '', text)  # Remove leading non-alphanumeric characters
        cleaned = re.sub(r'[^\w]+$', '', cleaned)  # Remove trailing non-alphanumeric characters
        return cleaned
    else:
        return text  # If it's not a string, return the original value


def main(pdf_file_path_new,pdf_file_path_old):
    ##############################
    pdf_file_path = pdf_file_path_new
    headings_by_page, header_footer_extracted, number_of_levels = extract_headings_from_pdf(pdf_file_path)
    # Save results to df
    extracted_headings_with_y_coordinates_new = save_headings_to_df(headings_by_page)
    header_footer_texts_new = save_header_footer_to_df(header_footer_extracted)
    df = header_footer_texts_new
    # Filter out rows where the 'Level' column contains the word 'Body'
    header_footer_texts_new_filtered_df = df[df['Level'] != 'Body']

    text_between_headings_in_order_new_df = extract_text_between_headings(extracted_headings_with_y_coordinates_new, header_footer_texts_new_filtered_df, pdf_file_path)

    updated_text_between_headings_in_order_new_df = clear_matching_texts(text_between_headings_in_order_new_df)
    #############################
    # Example usage
    pdf_file_path = pdf_file_path_old
    headings_by_page, header_footer_extracted, number_of_levels = extract_headings_from_pdf(pdf_file_path)
    # Save results to df
    extracted_headings_with_y_coordinates_old_df = save_headings_to_df(headings_by_page)
    header_footer_texts_old_df = save_header_footer_to_df(header_footer_extracted)
    # Read the CSV file into a DataFrame
    df = header_footer_texts_old_df
    # Filter out rows where the 'Level' column contains the word 'Body'
    header_footer_texts_old_filtered_df = df[df['Level'] != 'Body']

    text_between_headings_in_order_old = extract_text_between_headings(extracted_headings_with_y_coordinates_old_df, header_footer_texts_old_filtered_df, pdf_file_path)

    updated_text_between_headings_in_order_old_df = clear_matching_texts(text_between_headings_in_order_old)

    ##Comparison using SBERT Model

    # code to check whether the Text Between Headings is the same as the Start Heading or End Heading in the text_between_headings_in_order_old.csv file, and if so, remove that text.

    new_df = updated_text_between_headings_in_order_new_df.copy()
    old_df = updated_text_between_headings_in_order_old_df.copy()

    # Clean up the data: Convert non-string entries to empty strings or fill NaN values
    new_df['Start Heading'] = new_df['Start Heading'].fillna('').astype(str)
    new_df['Text Between Headings'] = new_df['Text Between Headings'].fillna('').astype(str)
    old_df['Start Heading'] = old_df['Start Heading'].fillna('').astype(str)
    old_df['Text Between Headings'] = old_df['Text Between Headings'].fillna('').astype(str)

    # Load pre-trained SBERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Helper function to combine 'Start Heading' and 'Text Between Headings' into one text for comparison
    def combine_text(df):
        return df['Start Heading'] + " " + df['Text Between Headings']

    # Combine the text in both new and old dataframes
    new_texts = combine_text(new_df).tolist()
    old_texts = combine_text(old_df).tolist()

    # Generate SBERT embeddings for both new and old text
    new_embeddings = model.encode(new_texts, convert_to_tensor=True)
    old_embeddings = model.encode(old_texts, convert_to_tensor=True)

    # Compute cosine similarity between new and old embeddings
    similarity_matrix = cosine_similarity(new_embeddings.cpu(), old_embeddings.cpu())

    # Find the closest matching rows based on the highest cosine similarity score
    closest_matches = []
    for new_index, new_row in new_df.iterrows():
        # Get the index of the most similar row in the old file
        most_similar_old_index = similarity_matrix[new_index].argmax()
        similarity_score = similarity_matrix[new_index][most_similar_old_index]

        # Retrieve corresponding rows from both files
        old_row = old_df.iloc[most_similar_old_index]

        # Store the matching rows with similarity score
        closest_matches.append({
            "New Start Heading": new_row['Start Heading'],
            "New Text Between Headings": new_row['Text Between Headings'],
            "Old Start Heading": old_row['Start Heading'],
            "Old Text Between Headings": old_row['Text Between Headings'],
            "Similarity Score": similarity_score
        })

    # Convert to a new dataframe
    closest_matching_rows_df = pd.DataFrame(closest_matches)
    ##Using Groq llm model shared by Abhishek
    ##Drop rows with similarity score >= 1
    closest_matching_rows_filtered_df = closest_matching_rows_df[closest_matching_rows_df['Similarity Score'] < 1]

    #############
    df = closest_matching_rows_filtered_df
    # Apply the function to each row of the DataFrame
    added_deleted_results = df.apply(
        lambda row: find_added_deleted_with_groq(
            str(row['Old Start Heading']) + ' ' + str(row['Old Text Between Headings']),
            str(row['New Start Heading']) + ' ' + str(row['New Text Between Headings'])
        ),
        axis=1
    )
    # Create new columns in the DataFrame
    df['Added Text'], df['Deleted Text'],  df['Change_summary'], df['Impact'],df['JSON Response'] = zip(*added_deleted_results.apply(parse_response))
    # differences_with_added_deleted_explanation_groq_df = df

    ######################## trying to exract json response using below code

    # df = differences_with_added_deleted_explanation_groq_df
    # Extract texts and create new columns
    df['Added Text'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'added_text', 'deleted_text'))
    df['Deleted Text'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'deleted_text', 'Change_summary'))
    df['Change_summary'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'Change_summary', 'Impact'))
    df['Impact'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'Impact', 'json_end'))
    # modified_differences_with_explanation_df = df
    #############code is able to extract properly irrespctive of json format.

    ##extracting only numbers in impact column

    # df = modified_differences_with_explanation_df
    # Apply the function to the 'Impact' column
    df['Impact'] = df['Impact'].apply(lambda x: extract_numbers(str(x)))
    # filtered_impact_df = df

    #dropping rows with impact score 1
    # df = filtered_impact_df
    # Drop rows where 'Impact' contains '1' or '1.0' (including variations)
    filtered_differences_df = df[~df['Impact'].astype(str).str.contains(r'^\s*(1|1\.0)\s*(\(.*\))?$', na=False, case=False)]

    # removing symobols before and after text in columns
    df = filtered_differences_df

    # Apply the cleaning function to the specified columns
    columns_to_clean = ['Added Text', 'Deleted Text', 'Change_summary']
    for column in columns_to_clean:
        df[column] = df[column].apply(clean_text)
    # cleaned_filtered_differences_df = df

    ########################

    # mapping the page numebers of the headings in the new pdf file.

    extracted_headings_df = extracted_headings_with_y_coordinates_new
    cleaned_differences_df = df
    # Create a dictionary to map 'Text' to 'Page Number' from extracted_headings_df
    text_to_page_map = dict(zip(extracted_headings_df['Text'], extracted_headings_df['Page Number']))
    # Map the 'New Start Heading' column in cleaned_differences_df to the new page number
    cleaned_differences_df['New Page Number'] = cleaned_differences_df['New Start Heading'].map(text_to_page_map)
    # updated_cleaned_filtered_differences_df = cleaned_differences_df

    #mapping of the page numbers in headings in old pdf file

    extracted_headings_df = extracted_headings_with_y_coordinates_old_df
    cleaned_differences_df = cleaned_differences_df
    # Create a dictionary to map 'Text' to 'Page Number' from extracted_headings_df
    text_to_page_map = dict(zip(extracted_headings_df['Text'], extracted_headings_df['Page Number']))
    # Map the 'New Start Heading' column in cleaned_differences_df to the new page number
    cleaned_differences_df['Old Page Number'] = cleaned_differences_df['Old Start Heading'].map(text_to_page_map)
    # updated_cleaned_filtered_differences_complete_df = cleaned_differences_df

    # dropping rows in which the Impact is blank
    df = cleaned_differences_df
    # Drop rows where 'Impact' is empty or NaN
    df_cleaned = df.dropna(subset=['Impact'])
    df_cleaned = df_cleaned[df_cleaned['Impact'].astype(bool)]  # This removes rows with blank strings
    cleaned_updated_cleaned_filtered_differences_complete = df_cleaned
    # Save the cleaned DataFrame back to a CSV file
    df_cleaned.dropna(inplace=True)
    summary = df_cleaned['Change_summary'].tolist()
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


    # df_cleaned.to_csv(csv_file_path, index=False)


    ##Highlighting in pdf new - working code

    # Download the necessary NLTK resources (only need to run once)
    nltk.download('punkt')
    df = cleaned_updated_cleaned_filtered_differences_complete

    # Load the PDF file
    pdf_file_path = 'new.pdf'
    doc = fitz.open(pdf_file_path)

    # Define the color green
    red_color = (0, 1, 0)  # RGB values (0, 1, 0) for green

    # Check if 'Added Text' and 'New Page Number' columns exist
    if 'Added Text' in df.columns and 'New Page Number' in df.columns:
        # Iterate over each entry in the DataFrame
        for index, row in df.iterrows():
            # Ensure 'Deleted Text' is a string
            if pd.isna(row['Added Text']):
                print(f"Row {index}: 'Added Text' is missing, skipping this row.")
                continue  # Skip rows where 'Deleted Text' is NaN

            # Tokenize the text into sentences
            sentences = sent_tokenize(str(row['Added Text']))  # Convert to string if not already

            # Get the specified page number (convert to zero-based index)
            specified_page_number = int(row['New Page Number']) - 1

            # Iterate over sentences
            for sentence in sentences:
                # Start searching from the specified page
                found = False
                for page_num in range(specified_page_number, doc.page_count):
                    page = doc.load_page(page_num)  # Load the page
                    text_instances = page.search_for(sentence)

                    if text_instances:  # If found, highlight and break
                        for inst in text_instances:
                            # Highlight the found sentence
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=red_color)  # Set highlight color
                            highlight.update()  # Update the highlight annotation
                        found = True
                        break  # Stop searching once the sentence is found

                # Optionally, you can print or log if a sentence wasn't found
                if not found:
                    print(f"Sentence not found: '{sentence}' on page {specified_page_number+1}")

    # Save the modified PDF
    modified_pdf_path_new = 'new_highlighted.pdf'
    doc.save(modified_pdf_path_new)
    doc.close()

    print(f'Highlighted sentences have been saved to {modified_pdf_path_new}')

    ##Highlighting in pdf old - working code

    # Download the necessary NLTK resources (only need to run once)
    nltk.download('punkt')

    df = cleaned_updated_cleaned_filtered_differences_complete

    # Load the PDF file
    pdf_file_path = 'old.pdf'
    doc = fitz.open(pdf_file_path)

    # Define the color green
    red_color = (1, 0, 0)  # RGB values (0, 1, 0) for green

    # Check if 'Added Text' and 'New Page Number' columns exist
    if 'Deleted Text' in df.columns and 'Old Page Number' in df.columns:
        # Iterate over each entry in the DataFrame
        for index, row in df.iterrows():
            # Ensure 'Deleted Text' is a string
            if pd.isna(row['Deleted Text']):
                print(f"Row {index}: 'Deleted Text' is missing, skipping this row.")
                continue  # Skip rows where 'Deleted Text' is NaN

            # Tokenize the text into sentences
            sentences = sent_tokenize(str(row['Deleted Text']))  # Convert to string if not already

            # Get the specified page number (convert to zero-based index)
            specified_page_number = int(row['Old Page Number']) - 1

            # Iterate over sentences
            for sentence in sentences:
                # Start searching from the specified page
                found = False
                for page_num in range(specified_page_number, doc.page_count):
                    page = doc.load_page(page_num)  # Load the page
                    text_instances = page.search_for(sentence)

                    if text_instances:  # If found, highlight and break
                        for inst in text_instances:
                            # Highlight the found sentence
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=red_color)  # Set highlight color
                            highlight.update()  # Update the highlight annotation
                        found = True
                        break  # Stop searching once the sentence is found

                # Optionally, you can print or log if a sentence wasn't found
                if not found:
                    print(f"Sentence not found: '{sentence}' on page {specified_page_number+1}")

    # Save the modified PDF
    modified_pdf_path_old = 'old_highlighted.pdf'
    doc.save(modified_pdf_path_old)
    doc.close()

    print(f'Highlighted sentences have been saved to {modified_pdf_path_old}')

    return (modified_pdf_path_new, modified_pdf_path_old, summary, impact_level)