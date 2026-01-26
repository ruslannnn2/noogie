# noogie - News through a different lens

Best Design at PennApps XXVI🏆

!["preview"](assets/noogie.png "Preview")
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![React](https://img.shields.io/badge/-React-61DAFB?style=flat-square&logo=react&logoColor=white)
![Flask](https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask&logoColor=white)
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/-HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/-CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white)
![OpenAPI](https://img.shields.io/badge/OpenAPI-6BA539?logo=openapiinitiative&logoColor=white)
![Emacs](https://img.shields.io/badge/Emacs-%237F5AB6.svg?&logo=gnu-emacs&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3FCF8E?logo=supabase&logoColor=fff)
![ETL](https://custom-icon-badges.demolab.com/badge/ETL-9370DB?logo=etl-logo&logoColor=fff)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=fff)

A project built to collect, duplicate, cluster, and summarize breaking news stories from across the web to mitigate biases that different news sources give. 

## Overview

DEMO - https://www.youtube.com/watch?v=rZpY3STlQxs

noogie automatically fetches articles from major outlets, groups similar stories, and generates concise AI-powered headlines.
It helps users see the big picture of what’s happening, filtering the opinions of different news sources.
!["preview"](assets/fullview.png "Preview")


## Features

* Multi-Source Collection – Pulls stories via RSS and scraping from a variety of news sources
* Clean Parsing – Uses newspaper4k to extract titles and content, filtering out empty or paywalled pages.
* Smart Deduplication – Computes sentence embeddings (sentence-transformers) and stores them in a FAISS vector index
* Clustering & Summarization – Groups similar articles and calls OpenAI/Gemini to generate short, human-readable cluster headlines
* Scalable Design – Supports multithreading (planned) so multiple feeds can be fetched concurrently.
!["preview"](assets/nodes.png "Preview")

## Tech Stack
* TypeScript + React with `d3.js` for node visualization
* Python `newspaper4k` to extract news sources & `flask` for the servers
* OpenAI's GPT 4o-mini for summaries
* Supabase for storage

## Usage
``` 
cd front-end
npm install
npm run dev
```

```
cd server
python -m venv
pip install -r requirements.txt
python server.py
python main.py
```
## API Endpoints
- `GET` api/clusters
- `GET` api/articles
- `GET` api/clusters/:clusters_id/articles
- `GET` api/clusters/:cluster_id
- `GET` api/articles/:article_id
- `POST` api/clusters/batch
- `POST` api/data/bulk
- `POST` api/clusters/:cluster_id/articles/batch


## Acknowledgements
A special thanks to PennApps 2025.


## License 
Copyright (c) 2025 Jatin Punjabi, Andres Lopez, Ruslan Akmyradov
