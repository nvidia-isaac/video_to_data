# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Test visual extractor on sample video.

This is a manual integration test -- run it directly via:
    python tests/test_visual_extractor.py <video_path>
"""

import logging
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_ingestion_agent.ingestion.entity_graph.extractors import VisualExtractor
from video_ingestion_agent.utils.vector_database import VectorDatabase
from video_ingestion_agent.utils.video_processor import VideoProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="Manual integration test -- requires a video_path CLI argument")
def test_visual_extractor(video_path: str, output_dir: str = "outputs/test"):
    """
    Test visual extractor end-to-end.

    Args:
        video_path: Path to test video
        output_dir: Directory for outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Testing visual extractor on: {video_path}")

    # 1. Extract frames
    logger.info("\n=== Step 1: Extract Frames ===")
    processor = VideoProcessor(video_path)
    metadata = processor.get_metadata()

    logger.info(f"Video duration: {metadata.duration:.1f}s")
    logger.info(f"Video resolution: {metadata.width}x{metadata.height}")
    logger.info(f"Video FPS: {metadata.fps:.1f}")

    frames = list(processor.extract_frames(fps=1.0))
    logger.info(f"Extracted {len(frames)} frames")

    # 2. Initialize visual extractor
    logger.info("\n=== Step 2: Initialize Visual Extractor ===")
    extractor = VisualExtractor(
        vlm_model="nvidia/Cosmos-Reason2-8B",
        embedding_model="google/siglip2-base-patch16-256",
        device="cuda",
        chunk_size=30.0,
        chunk_overlap=5.0,
    )

    # 3. Extract embeddings
    logger.info("\n=== Step 3: Extract Frame Embeddings ===")
    embeddings = extractor.extract_embeddings(frames, batch_size=32)

    logger.info(f"Extracted {len(embeddings)} embeddings")
    if embeddings:
        logger.info(f"Embedding dimension: {embeddings[0][1].shape[0]}")

    # 4. Store in vector database
    logger.info("\n=== Step 4: Store in Vector Database ===")
    db_path = output_dir / "test_vector.db"
    vector_db = VectorDatabase(str(db_path), embedding_dim=768)

    # Add video
    video_id = Path(video_path).stem
    vector_db.add_video(
        video_id=video_id,
        video_path=video_path,
        duration=metadata.duration,
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
    )

    # Add frame embeddings
    frame_data = []
    for frame, (frame_id, embedding) in zip(frames, embeddings, strict=False):
        frame_data.append(
            {
                "frame_id": frame_id,
                "video_id": video_id,
                "timestamp": frame.timestamp,
                "embedding": embedding,
                "metadata": frame.metadata,
            }
        )

    vector_db.add_frames_batch(frame_data)
    logger.info(f"Stored {len(frame_data)} frames in vector database")

    # 5. Generate captions
    logger.info("\n=== Step 5: Generate Captions ===")
    captions = extractor.extract_captions(frames, metadata.duration)

    logger.info(f"Generated {len(captions)} captions")

    # Save captions
    captions_file = output_dir / f"{video_id}_captions.txt"
    with open(captions_file, "w") as f:
        for i, caption in enumerate(captions, 1):
            f.write(f"=== Caption {i} ===\n")
            f.write(f"Time: [{caption.start_t:.1f}s - {caption.end_t:.1f}s]\n")
            f.write(f"Frames: {len(caption.chunk_frames)}\n")
            f.write(f"Location: {caption.location}\n")
            f.write(f"\n{caption.text}\n\n")

            # Also log to console
            logger.info(f"\nCaption {i} [{caption.start_t:.1f}s - {caption.end_t:.1f}s]:")
            logger.info(f"Location: {caption.location}")
            logger.info(f"Text: {caption.text[:200]}...")

    logger.info(f"Captions saved to: {captions_file}")

    # 6. Test vector search
    logger.info("\n=== Step 6: Test Vector Search ===")

    # Search with first frame as query
    if embeddings:
        query_embedding = embeddings[0][1]
        results = vector_db.search(query_embedding=query_embedding, video_id=video_id, top_k=5)

        logger.info("Top 5 similar frames to frame 0:")
        for result in results:
            logger.info(
                f"  Frame {result.frame_id} @ {result.timestamp:.1f}s: "
                f"similarity={result.similarity:.3f}"
            )

    # 7. Summary
    logger.info("\n=== Summary ===")
    logger.info(f"✓ Processed video: {video_path}")
    logger.info(f"✓ Extracted {len(frames)} frames")
    logger.info(f"✓ Generated {len(embeddings)} embeddings")
    logger.info(f"✓ Created {len(captions)} captions")
    logger.info(f"✓ Vector DB: {db_path}")
    logger.info(f"✓ Captions: {captions_file}")

    logger.info("\n✅ Visual extractor test complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test visual extractor")
    parser.add_argument("video_path", help="Path to test video")
    parser.add_argument("--output-dir", default="outputs/test", help="Output directory")

    args = parser.parse_args()

    test_visual_extractor(args.video_path, args.output_dir)
