import os
import sys
import glob
import asyncio
import httpx

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000/api/v1")


async def ingest_folder(folder_path: str):
    if not os.path.exists(folder_path):
        print(f"[Error] Folder does not exist: {folder_path}")
        return

    # Find supported documents
    extensions = ["*.pdf", "*.txt", "*.md", "*.markdown"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))

    if not files:
        print(f"[Notice] No PDF, TXT, or MD files found in '{folder_path}'.")
        return

    print(f"\n🚀 Found {len(files)} files to ingest into Production RAG:")
    async with httpx.AsyncClient(timeout=120.0) as client:
        for idx, file_path in enumerate(files, start=1):
            filename = os.path.basename(file_path)
            doc_id = os.path.splitext(filename)[0]
            print(f"[{idx}/{len(files)}] Ingesting '{filename}'...", end=" ", flush=True)

            try:
                with open(file_path, "rb") as f:
                    response = await client.post(
                        f"{API_URL}/ingest/file",
                        files={"file": (filename, f, "application/octet-stream")},
                        data={"document_id": doc_id}
                    )
                if response.status_code == 200:
                    data = response.json()
                    print(f"✔ Done ({data['total_chunks']} chunks, {data['time_taken_ms']}ms)")
                else:
                    print(f"❌ Failed ({response.status_code}): {response.text}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    print("\n✨ Ingestion complete! Open http://localhost:8000 to test queries and inspect metrics.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_folder.py <path_to_folder>")
        sys.exit(1)
    asyncio.run(ingest_folder(sys.argv[1]))
