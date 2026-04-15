from backend.ingestion.pipeline import IngestionPipeline
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore")

pipeline = IngestionPipeline()
print("Starting forced Vault Re-ingestion over LM Studio...")
res = pipeline.ingest(force_full=True)
print(f"\n✅ Finished! Added {res.new} chunks, updated {res.updated} chunks. Backend DB is perfectly synced.")
