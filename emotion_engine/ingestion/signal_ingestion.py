"""
Signal Ingestion Layer: Accepts and validates input signals.
"""

class SignalIngestion:
    def ingest(self, text: str, metadata: dict, context: dict) -> dict:
        """
        Validate and structure input for downstream processing.
        """
        # Defensive: Ensure required fields
        if not text or not isinstance(metadata, dict) or not isinstance(context, dict):
            raise ValueError("Invalid input for ingestion.")
        return {"text": text, "metadata": metadata, "context": context}
