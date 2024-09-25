import os
import shutil
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd

class DocumentOrganizer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.document_path = os.path.join(base_path, 'documents')
        self.metadata_path = os.path.join(base_path, 'metadata')
        self.index_path = os.path.join(base_path, 'index')

        # Ensure directories exist
        for path in [self.document_path, self.metadata_path, self.index_path]:
            os.makedirs(path, exist_ok=True)

    def add_document(self, file_path, metadata_path):
        # Generate a unique identifier
        doc_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file_path)}"

        # Determine file type and copy to appropriate directory
        _, ext = os.path.splitext(file_path)
        if ext.lower() in ['.txt', '.pdf']:
            dest_dir = os.path.join(self.document_path, ext[1:])
        elif ext.lower() in ['.png', '.jpg', '.jpeg']:
            dest_dir = os.path.join(self.document_path, 'images')
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, doc_id)
        shutil.copy2(file_path, dest_path)

        # Process and save metadata
        _, meta_ext = os.path.splitext(metadata_path)
        if meta_ext.lower() == '.csv':
            self._process_csv_metadata(metadata_path, doc_id)
        elif meta_ext.lower() == '.xml':
            self._process_xml_metadata(metadata_path, doc_id)
        else:
            raise ValueError(f"Unsupported metadata format: {meta_ext}")

        return doc_id

    def _process_csv_metadata(self, csv_path, doc_id):
        df = pd.read_csv(csv_path)
        metadata = df.to_dict(orient='records')[0]  # Assuming one row per document
        metadata['document_id'] = doc_id
        metadata['date_added'] = datetime.now().isoformat()

        metadata_path = os.path.join(self.metadata_path, 'csv', f"{doc_id}.csv")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        pd.DataFrame([metadata]).to_csv(metadata_path, index=False)

    def _process_xml_metadata(self, xml_path, doc_id):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if 'EAD' in root.tag:
            metadata = self._process_ead_xml(root)
        elif 'record' in root.tag:
            metadata = self._process_marc_xml(root)
        else:
            raise ValueError("Unsupported XML format. Expected EAD or MARC XML.")

        metadata['document_id'] = doc_id
        metadata['date_added'] = datetime.now().isoformat()

        metadata_path = os.path.join(self.metadata_path, 'xml', f"{doc_id}.xml")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        tree.write(metadata_path)

    def _process_ead_xml(self, root):
        metadata = {}
        metadata['title'] = root.find('.//unittitle').text if root.find('.//unittitle') is not None else ''
        metadata['date'] = root.find('.//unitdate').text if root.find('.//unitdate') is not None else ''
        # Add more EAD-specific metadata extraction as needed
        return metadata

    def _process_marc_xml(self, root):
        metadata = {}
        for datafield in root.findall('.//datafield'):
            tag = datafield.get('tag')
            if tag == '245':  # Title
                metadata['title'] = datafield.find('.//subfield[@code="a"]').text
            elif tag == '100':  # Author
                metadata['author'] = datafield.find('.//subfield[@code="a"]').text
            # Add more MARC-specific metadata extraction as needed
        return metadata

    def get_document_path(self, doc_id):
        for root, _, files in os.walk(self.document_path):
            if doc_id in files:
                return os.path.join(root, doc_id)
        return None

    def get_metadata(self, doc_id):
        csv_path = os.path.join(self.metadata_path, 'csv', f"{doc_id}.csv")
        xml_path = os.path.join(self.metadata_path, 'xml', f"{doc_id}.xml")

        if os.path.exists(csv_path):
            return pd.read_csv(csv_path).to_dict(orient='records')[0]
        elif os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            if 'EAD' in root.tag:
                return self._process_ead_xml(root)
            elif 'record' in root.tag:
                return self._process_marc_xml(root)
        return None

# Usage example
organizer = DocumentOrganizer('/path/to/data')

# Adding a new document with CSV metadata
doc_id_csv = organizer.add_document('/path/to/sample.txt', '/path/to/metadata.csv')

# Adding a new document with EAD XML metadata
doc_id_ead = organizer.add_document('/path/to/sample.pdf', '/path/to/ead_metadata.xml')

# Adding a new document with MARC XML metadata
doc_id_marc = organizer.add_document('/path/to/sample.jpg', '/path/to/marc_metadata.xml')

# Retrieving document path and metadata
for doc_id in [doc_id_csv, doc_id_ead, doc_id_marc]:
    doc_path = organizer.get_document_path(doc_id)
    metadata = organizer.get_metadata(doc_id)
    print(f"Document ID: {doc_id}")
    print(f"Document path: {doc_path}")
    print(f"Metadata: {metadata}")
    print("---")
