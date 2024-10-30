Note: Use Markdown Cheat Sheet if you need more functionality
https://www.markdownguide.org/cheat-sheet/
### Date: Sep 24 2024 
- Topics of discussion
    - Concatenating metadata with documents
    - Repo organization
    - Data diagram:

![WhatsApp Image 2024-09-24 at 19 53 12_a203f424](https://github.com/user-attachments/assets/7868fbcc-1f64-4a4b-b408-81b515e24889)


- Action Items:
    * [x] Try and create manual questions (around 20-30)
    * [ ] Add data diagram to final report
    * [x] Add metadata to vector store
    * [x] Determine if FIASS can handle passing metadata; if not explore deeplake
    * [x] Add API code + parser
    * [x] Continue research; add to introduction, identify additional relevant papers
    * [ ] fix folder structure; add test files for each component, utils; test script for every component
    * [x] convert pdfs to text files; combine into csv (two halves due to size)


---
### Date: oct 1 2024 
- Topics of discussion
    - Discuss chunking sizes/overlap
    - What is expected with/in component test files

- Action Items:
* [x] try llama3
* [ ] decouple main py file for test files; run on subsets; demonstrate functionality
* [ ] how to link csv data to xml (Call number - Paul will figure this out)
* [x] Ask Amir for RAG papers
* [ ] Paul - test query for document metadata ONLY



---
### Date: oct 8 2024 
- Topics of discussion
    - Mid-point Presentation

- Action Items:
* [ ] make sure chunking is right; check success rates by top_k (need to integrate all data)
* [ ] try closed LLM generator + titan embeddings
* [ ] test scripts

---
### Date: oct 15 2024 
- Topics of discussion
    - test scripts and accuracies of top_k, chunk size
    - removal of timestamps from docs
    - generator tokenization max_limits

- Action Items:
* [-] Add in Bedrock + Titan as options for user
* [x] Two tables for accuracy on 13 questions for the 46 test docs
    * [x] One table comparing the generated responses for each model (Llama3.2, T5, Bedrock+Titan), for each of the questions
        * [ ] Include metadata retrieval
    * [x] One table showing accuracy of each model (target response vs predicted response)
* [-] Develop plan for Advanced RAG


---
### Date: oct 22 2024 
- Topics of discussion
    - titan/sonnet access; email sent to IT for help/access
    - fix top_k in retrieval evaluation
        - chunk accuracy matters less the document accuracy

- Action Items:
* [x] Add in Bedrock + Titan as options for user, pending IT
* [x] Clean tables in report showing accuracies for retrieval/generation
    * [x] Include things like: question, context, vectorstore output, prompt, generated output
* [ ] Multiple vectorstores - one with and one without metadata for comparing
* [x] Develop plan for Advanced RAG; pick ideas and incorporate into our draft(s).

![SampleData_AccEval_table](https://github.com/user-attachments/assets/5a92c996-5edf-486c-92b5-74b9a0603c0d)

![SampleData_AccEval_linecharts](https://github.com/user-attachments/assets/c35b960e-cf49-4a51-8dfe-a83628e8c79c)



---
### Date: oct 29 2024 
- Topics of discussion
    - Reranking not making much of a difference, if at all, on early sample testing. It seems the top_k documents retrieved are often very similar if not identitally ordered. Instructor as original retriever yields better reranking results than miniLM, as expected. Bge returned identical retrieved docs and accuracies compared to the original instructor retrieved docs, and it cut off the lowest scoring doc when top=3. Qwen often retreived the same or similar documents as the original retriever but often extracted the wrong 'best' match, indicating the similarity scoring mechanism here might require additional analysis.
    - Qwen seems better at getting all the docs while bge is better at getting the 'best' match; try multiple steps in reranking to first to Qwen and the bge
    - Given the lack of promise, worth continuing to pursue???
    - ideas: gets relevance scores from vectorstore and then enhance reranker using those scores. Reference
        - similarity_search_with_relevance_scores instead of just similarity_search
        - NodeWithScore scores in nodes at:                       https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/
        - incorporate alternate similarity scores

- Action Items:
* [ ] Retry retrieval accuracies on 100 questions instead of 13. If same accuracies then stop using reranking.
* [ ] Consider summarization (summarize each doc with bedrock, put into seperate vector store)
* [ ] Create multiple vector stores (for chunk size 250 and 1000, and with metadata and without metadata)
* [ ] 























---
### Date: sep 24 2024 
- Topics of discussion




- Action Items:
* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
