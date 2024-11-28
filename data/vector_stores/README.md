# Vector Stores Directory

This directory was initially intended to store vector stores for the Retrieval-Augmented Generation (RAG) system. However, due to the large size of these files, keeping them in the repository is impractical.

## Best Practices for Vector Stores

- **Storage**: Vector stores should not be stored in this repository. Instead:
  - Use a **virtual machine (VM)** or **local machine** for storage and processing.
  - Alternatively, consider cloud storage solutions such as AWS S3, Google Cloud Storage, or Azure Blob Storage for hosting large files.
- **Loading**: Ensure your RAG system is set up to load vector stores from the designated machine or cloud storage.

## Why Not Store Vector Stores in the Repo?

1. **File Size**: Vector stores can be several gigabytes or larger, exceeding the recommended size limits for Git repositories.
2. **Performance**: Large files in a repository can significantly slow down cloning, pushing, and pulling operations.
3. **Version Control**: Git is not optimized for handling large binary files, leading to inefficient version control.

## Recommended Workflow

1. Store vector stores locally or on a VM where the RAG system is running.
2. Use environment variables or a configuration file to specify the path to the vector stores.
3. If needed, include a script to download vector stores from a cloud storage solution to the desired location.

By following these best practices, we can maintain the performance and efficiency of this repository while ensuring proper handling of vector store files.
