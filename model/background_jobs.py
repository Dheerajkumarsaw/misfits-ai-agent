"""
Background jobs for pre-computation tasks
Handles similar events computation with temporal filtering
"""

import threading
import time
from datetime import datetime
from typing import List
import json


class BackgroundJobs:
    """
    Manages background jobs for the recommendation system

    Features:
    - Similar events pre-computation (every 6 hours)
    - Temporal filtering (only future events as similar)
    - Graceful error handling
    - Thread-safe operations
    """

    def __init__(self, bot_instance):
        """
        Initialize background jobs

        Args:
            bot_instance: MeetupBot instance with chroma_manager
        """
        self.bot = bot_instance
        self.chroma_manager = bot_instance.chroma_manager
        self.running = False
        self._thread = None

        print("✅ BackgroundJobs initialized")

    def start_similar_events_job(self, interval_hours: int = 6):
        """
        Start background job to compute similar events

        Args:
            interval_hours: Hours between computation runs
        """
        if self.running:
            print("⚠️ Similar events job already running")
            return

        self.running = True
        self._thread = threading.Thread(
            target=self._similar_events_worker,
            args=(interval_hours,),
            daemon=True
        )
        self._thread.start()
        print(f"✅ Similar events job started (runs every {interval_hours} hours)")

    def stop_similar_events_job(self):
        """Stop the background job"""
        self.running = False
        if self._thread:
            print("🛑 Stopping similar events job...")

    def _similar_events_worker(self, interval_hours: int):
        """
        Worker thread for similar events computation

        Args:
            interval_hours: Hours between runs
        """
        # Run immediately on startup
        try:
            print("🔄 Running initial similar events computation...")
            self._compute_similar_events_with_temporal_filter()
        except Exception as e:
            print(f"⚠️ Initial similar events computation failed: {e}")

        # Then run periodically
        while self.running:
            try:
                # Sleep first
                sleep_seconds = interval_hours * 60 * 60
                print(f"⏰ Next similar events computation in {interval_hours} hours")
                time.sleep(sleep_seconds)

                if not self.running:
                    break

                # Run computation
                print("🔄 Computing similar events...")
                self._compute_similar_events_with_temporal_filter()
                print("✅ Similar events computation completed")

            except Exception as e:
                print(f"❌ Similar events job failed: {e}")
                import traceback
                traceback.print_exc()

    def _compute_similar_events_with_temporal_filter(self):
        """
        Compute similar events with temporal filtering

        For each event X with date D:
        - Find similar events using vector similarity
        - Only include events with date >= D (future events)
        - Store top 5 similar event IDs in metadata
        """
        try:
            # Get all events from ChromaDB
            all_events = self.chroma_manager.collection.get(
                include=['embeddings', 'metadatas', 'documents']
            )

            if not all_events or not all_events.get('ids'):
                print("⚠️ No events found in ChromaDB for similar events computation")
                return

            total_events = len(all_events['ids'])
            print(f"📊 Computing similar events for {total_events} events...")

            successful = 0
            failed = 0

            # Process each event
            for i, event_id in enumerate(all_events['ids']):
                try:
                    event_metadata = all_events['metadatas'][i]
                    event_embedding = all_events['embeddings'][i]

                    # Parse event date
                    event_date = self._parse_event_date(
                        event_metadata.get('start_time', '')
                    )

                    # Query similar events using vector similarity
                    similar_results = self.chroma_manager.collection.query(
                        query_embeddings=[event_embedding],
                        n_results=min(50, total_events)  # Get candidates for filtering
                    )

                    # TEMPORAL FILTER: Only events >= current event's date
                    future_similar_ids = []

                    if similar_results and similar_results.get('ids'):
                        for j, similar_id in enumerate(similar_results['ids'][0]):
                            # Skip self
                            if similar_id == event_id:
                                continue

                            # Get similar event metadata
                            similar_metadata = similar_results['metadatas'][0][j]

                            # Parse similar event date
                            similar_date = self._parse_event_date(
                                similar_metadata.get('start_time', '')
                            )

                            # Only include if similar event is ON or AFTER this event
                            if similar_date >= event_date:
                                future_similar_ids.append(similar_id)

                            # Stop when we have 5 future similar events
                            if len(future_similar_ids) >= 5:
                                break

                    # Update metadata with similar event IDs
                    updated_metadata = {
                        **event_metadata,
                        'similar_event_ids': json.dumps(future_similar_ids),
                        'similar_computed_at': datetime.now().isoformat()
                    }

                    self.chroma_manager.collection.update(
                        ids=[event_id],
                        metadatas=[updated_metadata]
                    )

                    successful += 1

                    # Log progress every 100 events
                    if (i + 1) % 100 == 0:
                        print(f"ca {i + 1}/{total_events} events processed")

                except Exception as e:
                    failed += 1
                    if failed <= 5:  # Only print first 5 errors
                        print(f"⚠️ Failed to compute similar events for {event_id}: {e}")

            print(f"✅ Similar events computation complete:")
            print(f"   - Successful: {successful}/{total_events}")
            print(f"   - Failed: {failed}/{total_events}")

        except Exception as e:
            print(f"❌ Critical error in similar events computation: {e}")
            import traceback
            traceback.print_exc()

    def _parse_event_date(self, start_time_str: str) -> datetime:
        """
        Parse event date from start_time string

        Args:
            start_time_str: Start time string (format: "2025-12-12 18:00 IST")

        Returns:
            Datetime object (defaults to current time if parsing fails)
        """
        try:
            # Handle format: "2025-12-12 18:00 IST"
            if ' IST' in start_time_str:
                date_part = start_time_str.split(' IST')[0].strip()
            else:
                date_part = start_time_str.strip()

            # Try parsing with different formats
            for fmt in ['%Y-%m-%d %H:%M', '%Y-%m-%d', '%d-%m-%Y %H:%M', '%d-%m-%Y']:
                try:
                    return datetime.strptime(date_part, fmt)
                except ValueError:
                    continue

            # If all parsing fails, return current time
            return datetime.now()

        except Exception:
            # Fallback to current time if parsing completely fails
            return datetime.now()

    def run_similar_events_once(self):
        """
        Run similar events computation once (synchronously)
        Useful for manual triggering or testing
        """
        print("🔄 Running similar events computation (manual trigger)...")
        try:
            self._compute_similar_events_with_temporal_filter()
            print("✅ Manual similar events computation completed")
            return True
        except Exception as e:
            print(f"❌ Manual similar events computation failed: {e}")
            return False
