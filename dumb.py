import chromadb
client = chromadb.PersistentClient(path=r"embedding\chroma_db\bge_m3\db_64")
print(client.list_collections())