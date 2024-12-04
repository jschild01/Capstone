import os
import time
import requests
import pandas as pd
import numpy as np

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
    request_pause = 1
    long_request_pause = 60
    
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
        time.sleep(1)
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
    
def filter_and_format_files(file_list_df: pd.DataFrame, mimetypes=None, w=0, h=0, iiif_output_format='jpg') -> pd.DataFrame:
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



# XML Scraper ----------------------------------------------------------------------------------------------------------
def get_url_xml(url): # rate limit(s) are 10 requests per minute
    # raw xml
    raw_xml = getXML(url)
    time.sleep(random.randint(7, 10))

    # raw uncleaned text inside the xml
    raw_text = getXMLText(url)
    time.sleep(random.randint(7, 10))
    return raw_xml, raw_text

def get_catalog_record_url(url):
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the XML content
        root = ET.fromstring(response.content)
        namespace = {'ead': 'http://ead3.archivists.org/schema/'}
        ref_element = root.find('.//ead:controlnote[@id="lccnNote"]/ead:p/ead:ref', namespace)

        # Extract the URL
        if ref_element is not None:
            catalog_record_url = ref_element.get('href')
            catalog_record_url = catalog_record_url + "/marcxml"
            return catalog_record_url
        else:
            catalog_record_url = f"Failed to fetch XML content. Status code: {response.status_code}"
    else:
        catalog_record_url = f"Failed to fetch XML content. Status code: {response.status_code}"

    # random sleep to avoid getting blocked/detected
    time.sleep(random.randint(1, 3))
    return catalog_record_url

def fetchXML(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup

def getXML(url):
    soup = fetchXML(url)
    return soup.prettify()

def getXMLText(url):
    soup = fetchXML(url)
    return soup.get_text()

def getXMLLinks(url):
    soup = fetchXML(url)
    links = []
    for link in soup.find_all("a"):
        links.append(link.get("href"))
    return links

def get_lccn(url):
    lccn = ''.join(filter(str.isdigit, url))
    return lccn

def get_collection(url):
    collection = url.split("/")[-3]
    return collection

def get_digital_url(row):
    baseurl = row.get("digital_content_baseurl")
    baseurl2 = row.get("digital_content_baseurl2")
    
    if pd.notna(row.get("marc_xml")) and pd.notna(baseurl):
        match = pd.Series(row["marc_xml"]).str.extract(rf'({baseurl}[^ ]*)')
        if not match.isna().values.any():
            return match[0].values[0]
    
    if pd.notna(row.get("ead_xml")) and pd.notna(baseurl):
        match = pd.Series(row["ead_xml"]).str.extract(rf'({baseurl}[^ ]*)')
        if not match.isna().values.any():
            return match[0].values[0]
    
    if pd.notna(row.get("marc_xml")) and pd.notna(baseurl2):
        match = pd.Series(row["marc_xml"]).str.extract(rf'({baseurl2}[^ ]*)')
        if not match.isna().values.any():
            return match[0].values[0]

    if pd.notna(row.get("ead_xml")) and pd.notna(baseurl2):
        match = pd.Series(row["ead_xml"]).str.extract(rf'({baseurl2}[^ ]*)')
        if not match.isna().values.any():
            return match[0].values[0]

    return "No url found"

def get_all_XML_urls(base_url):
    collections = getXMLLinks(base_url)
    collections_urls = [base_url + link for link in collections]

    # loop through each collection to get their corresponding years; not each collection has the same years available
    urls_master = []
    for collection_url in collections_urls:    
        
        # get the years for each collection and their urls
        years = getXMLLinks(collection_url)
        collectionYear_urls = [collection_url + year for year in years]

        # loop through each year to get the xml links
        for collectionYear_url in collectionYear_urls:
            xmlItems = getXMLLinks(collectionYear_url)
            full_urls = [collectionYear_url + item for item in xmlItems]
            urls_master.append(full_urls)
            time.sleep(random.randint(7, 10))
        
        time.sleep(random.randint(7, 10))

    # flatten master list so it is not a nested list
    urls_master = [item for sublist in urls_master for item in sublist]
    return urls_master

def fetch_ead_marc_xml():
    base_url = "https://findingaids.loc.gov/exist_collections/ead3master/"
    urls_master = get_all_XML_urls(base_url)
    df = pd.DataFrame(urls_master, columns=["urls"])

    # get additional data for the urls
    df["catalog_marcRecord_url"] = df["urls"].apply(get_catalog_record_url)
    df["ead_xml"], df["ead_xmlText"] = zip(*df["urls"].map(get_url_xml))
    df["marc_xml"], df["marc_xmlText"] = zip(*df["catalog_marcRecord_url"].map(get_url_xml))
    df["lccn"] = df["catalog_marcRecord_url"].apply(get_lccn)
    df["collection"] = df["urls"].apply(get_collection)

    # check for/if there is digital content
    df["digital_content_baseurl"] = df["collection"].apply(lambda x: f"http://hdl.loc.gov/loc.{x}/coll{x}")
    df["digital_content_baseurl2"] = df["collection"].apply(lambda x: f"https://hdl.loc.gov/loc.{x}/coll{x}")
    df["digital_content"] = df.apply( # check if digital content base url is present in the ead_xml or marc_xml (True/False)
                            lambda row: (row["digital_content_baseurl"] in row["ead_xml"]) or 
                                        (row["digital_content_baseurl"] in row["marc_xml"]) or
                                        (row["digital_content_baseurl2"] in row["ead_xml"]) or
                                        (row["digital_content_baseurl2"] in row["marc_xml"]),
                            axis=1
                        )

    # get full digital content url
    df["digital_content_fullurl"] = df.apply(get_digital_url, axis=1)
    return df