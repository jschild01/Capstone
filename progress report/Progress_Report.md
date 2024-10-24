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
* [ ] Add in Bedrock + Titan as options for user, pending IT
* [ ] Clean tables in report showing accuracies for retrieval/generation
    * [ ] Include things like: question, context, vectorstore output, prompt, generated output
* [ ] Multiple vectorstores - one with and one without metadata for comparing
* [ ] Develop plan for Advanced RAG; pick ideas and incorporate into our draft(s).










---
### Date: sep 24 2024 
- Topics of discussion




- Action Items:
* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
### Date: sep 24 2024 
- Topics of discussion




- Action Items:
* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
