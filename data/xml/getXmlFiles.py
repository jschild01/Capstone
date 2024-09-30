#%%
import pandas as pd 
import os
import re
import xml.etree.ElementTree as ET

# Load the xml master CSV file
df = pd.read_csv(r'C:\Users\schil\OneDrive\Desktop\School\Capstone\LOC\xml\xml_master3_afc.csv')

# Define column names that contain the XML
ead_xml_column = "ead_xml"
marc_xml_column = "marc_xml"

# Create directories to store the XML files
ead_output_dir = r"C:\Users\schil\OneDrive\Desktop\School\Capstone\LOC\xml\ead"
marc_output_dir = r"C:\Users\schil\OneDrive\Desktop\School\Capstone\LOC\xml\marc"
os.makedirs(ead_output_dir, exist_ok=True)
os.makedirs(marc_output_dir, exist_ok=True)

# Define a function to sanitize file names
def sanitize_filename(filename):
    # Remove invalid characters like newlines, spaces, and others
    return re.sub(r'[<>:"/\\|?*\n\r]', '', filename)

# Iterate over rows and save both EAD and MARC XML values to separate files
for index, row in df.iterrows():
    ead_xml_content = row[ead_xml_column]
    marc_xml_content = row[marc_xml_column]
    
    # Get file name from ead content
    root = ET.fromstring(ead_xml_content)
    namespaces = {'ead': 'http://ead3.archivists.org/schema/'}
    recordid = root.find('.//ead:control/ead:recordid', namespaces)
    instance_url = recordid.attrib.get('instanceurl')
    id_value = instance_url.split('.')[-1]

    # Define the EAD file name and save it
    ead_file_name = f"{id_value}_ead.xml"
    ead_file_path = os.path.join(ead_output_dir, sanitize_filename(ead_file_name))
    with open(ead_file_path, "w", encoding="utf-8") as ead_xml_file:
        ead_xml_file.write(ead_xml_content)

    # Define the MARC file name and save it
    marc_file_name = f"{id_value}_marc.xml"
    marc_file_path = os.path.join(marc_output_dir, sanitize_filename(marc_file_name))
    with open(marc_file_path, "w", encoding="utf-8") as marc_xml_file:
        marc_xml_file.write(marc_xml_content)

print(f"Saved all EAD and MARC XML files")















