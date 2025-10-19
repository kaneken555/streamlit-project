.PHONY: ingest app clean

ingest:
\tpython ingest.py

app:
\tstreamlit run app.py

clean:
\trm -rf chroma_db
