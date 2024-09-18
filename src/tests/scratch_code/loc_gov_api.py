import time
import sys
import os
import requests
import pandas as pd
import numpy as np

# Prompt user for input_search
input_search = input("Enter the Library of Congress search URL: ")

output_search_csv = 'search_results.csv'
output_file_csv = 'file_list.csv'

# we want [application/pdf], [video/mp4], [audio/mp3], [text/plain], [application/msword], [application/epub+zip]
target_file_formats = []
iiif_output_format = 'jpg'
iiif_width = 0
iiif_height = 0

request_pause = 1
long_request_pause = 60

def save_dataframe_to_csv(df: pd.DataFrame, csv_file: str, append=True):
    if os.path.isfile(csv_file) and append:
        old_df = pd.read_csv(csv_file)
        new_df = pd.concat([df,old_df], axis=0)
        new_df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, index=False, mode='w')
        
def flatten_nested_json(record, prefix=None, new_record=None):
    if new_record == None:
        new_record = {}
    if isinstance(record, dict):
        for key in record:
            if isinstance(record[key],list):
                if prefix is None:
                    for i,item in enumerate(record[key]):
                        new_record = flatten_nested_json(item, f'{key}.{i}', new_record)
                else:
                    for i,item in enumerate(record[key]):
                        new_record = flatten_nested_json(item, f'{prefix}.{key}.{i}', new_record)
            elif isinstance(record[key],dict):
                if prefix is None:
                    new_record = flatten_nested_json(record[key], key, new_record)
                else:
                    new_record = flatten_nested_json(record[key], f'{prefix}.{key}', new_record)
            else:
                if prefix is None:
                    new_record[key] = record[key]
                else:
                    new_record[f'{prefix}.{key}'] = record[key]
    else:
        new_record[prefix] = record

    return new_record

def fetch_api_data(url: str, attempt_num: int, params: dict):
    time.sleep(request_pause)
    url = url.replace('http:','https:')
    result = requests.get(url, params = params)

    if result.status_code == 429:
        time.sleep(long_request_pause)
        new_attempt_num = attempt_num + 1
        return fetch_api_data(url, attempt_num = new_attempt_num, params = params)
    elif (500 <= result.status_code) & (600 > result.status_code):
        if attempt_num < 5:
            print(f"Server error #{attempt_num}. Pausing 10 seconds and retrying.")
            time.sleep(10)
        elif attempt_num <= 15:
            print(f"Server error #{attempt_num}. Pausing 1 minute and retrying.")
            time.sleep(60)
        else:
            print(f"Server error ongoing (#{attempt_num}). Pausing and awaiting operator input to resume.")
            input("Press Enter to continue...")
        new_attempt_num = attempt_num + 1
        return fetch_api_data(url, attempt_num = new_attempt_num, params = params)
    elif result.status_code == 403:
        print(f"Received 403 status, skipping: {url}")
        return None

    try:
        output_json = result.json()
        if ('status' in output_json) and (output_json['status']==404):
            print(f"Last page empty, ending at: {url}")
            return None
        return output_json
    except Exception as e:
        print(f"Skipping {url}. Could not interpret as json. Error: {e}")
        return None

def retrieve_search_results(request_url: str) -> pd.DataFrame:
    all_results = []
    params = {'fo':'json', 'at': 'results,pagination', 'c': 100}
    i = 1
    while True:
        params.update({'sp':i})
        response = fetch_api_data(request_url, attempt_num=0, params = params)
        if not response:
            break
        all_results.extend(response['results'])
        i += 1
        time.sleep(request_pause)
    all_results = [r for r in all_results if 'item' in r['url']]
    all_results = [flatten_nested_json(r) for r in all_results]
    
    all_results_df = pd.DataFrame(all_results)
    return all_results_df

def extract_file_data_from_item(item_record: dict) -> pd.DataFrame:
    files_df = pd.DataFrame()

    try:
        if isinstance(item_record, dict):
            item_id = item_record.get('item', {}).get('id') or item_record.get('id')
        else:
            item_id = 'No ID available'

        resources = item_record.get('resources', [])
        
        for resource_num, resource in enumerate(resources):
            segments = resource.get('files', [])
            for segment_num, segment in enumerate(segments):
                df = pd.json_normalize(segment)
                df['resource_url'] = resource.get('url', 'No URL available')
                df['resource_num'] = resource_num
                df['segment_num'] = segment_num
                df['id'] = item_id
                try:
                    files_df = pd.concat([files_df, df], ignore_index=True)
                except Exception:
                    pass
            
            if 'pdf' in resource:
                pdf_url = resource['pdf']
                if 'url' in df.columns and len(df[df['url'] == pdf_url]) == 0:
                    pdf_item = {
                        'url': pdf_url,
                        'resource_url': resource.get('url', 'No URL available'),
                        'resource_num': resource_num,
                        'id': item_id,
                        'mimetype': 'application/pdf'
                    }
                    files_df = pd.concat(
                        [files_df, pd.DataFrame([pdf_item])],
                        ignore_index=True
                    )

        return files_df

    except Exception:
        return pd.DataFrame()
    
def filter_and_format_files(file_list_df: pd.DataFrame, mimetypes=None, w=0, h=0, iiif_output_format=iiif_output_format) -> pd.DataFrame:
    if mimetypes is None:
        mimetypes = []

    if len(mimetypes)>0:
        filtered_formats = file_list_df[file_list_df['mimetype'].isin(mimetypes)].copy()
    else:
        filtered_formats = file_list_df.copy()

    filtered_formats.rename(columns={
        'size':'source_size',
        'url':'source_url',
        'height':'source_height',
        'width':'source_width',
        'levels':'scaleFactor_levels'
        }, inplace=True)

    if w!=0:
        params = f"full/{w},/0/default.{iiif_output_format}"
    elif h!=0:
        params = f"full/,{h}/0/default.{iiif_output_format}"
    else:
        params = f"full/pct:100/0/default.{iiif_output_format}"

    if 'info' not in filtered_formats.keys():
        filtered_formats['info'] = None
    filtered_formats['iiif_url'] = np.where(
        (
            (filtered_formats['source_url'].str.contains('iiif'))&
            (filtered_formats['source_url'].str.contains('default.jpg'))
        ),
        filtered_formats['source_url'],
        None
    )
    filtered_formats['iiif_url'] = np.where(
        filtered_formats['info'].str.contains('iiif')==True,
        filtered_formats['info'].str.replace('info.json', params, regex=False),
        filtered_formats['iiif_url']
    )
    return filtered_formats

def standardize_dataframe_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = None
    return df[cols]

def process_item_data(row: pd.Series, index: int) -> pd.DataFrame:
    item_id = row['id']

    try:
        item_params = {'fo':'json','at':'item,resources'}
        item_record = fetch_api_data(item_id, attempt_num=0, params=item_params)
        if item_record is None:
            return pd.DataFrame(columns=output_columns)

        if isinstance(item_record, dict):
            item_record['id'] = item_id

        files = extract_file_data_from_item(item_record)

        if len(files) == 0:
            return pd.DataFrame(columns=output_columns)

        filtered_files = filter_and_format_files(files, mimetypes=target_file_formats, w=iiif_width, h=iiif_height)
        normalized_cols = standardize_dataframe_columns(filtered_files, output_columns)

        save_dataframe_to_csv(filtered_files, output_file_csv)
        return filtered_files
    
    except Exception:
        return pd.DataFrame(columns=output_columns)

output_columns = [
    'id',
    'mimetype',
    'iiif_url',
    'resource_url',
    'resource_num',
    'segment_num',
    'source_url',
    'source_size',
    'source_width',
    'source_height',
    'scaleFactor_levels',
    'info'
]

start_time = time.time()

input_data = retrieve_search_results(input_search)
save_dataframe_to_csv(input_data, csv_file=output_search_csv, append=False)

result = pd.concat([process_item_data(row,idx) for idx, row in input_data.iterrows()], ignore_index=True)

result.dropna(axis=0, how='all', inplace=True)
result.reset_index()

end_time = time.time()
total_time = end_time - start_time

print(f'Total number of downloadable files in your list: {len(result)}')
print(f'Total storage size required for full sized source files (for those files that provided storage size): {result["source_size"].sum()/1000000000} GB')
print('Breakdown of file formats:')
print(result['mimetype'].value_counts())

print('Your output files are:')
print(output_search_csv)
print(output_file_csv)
