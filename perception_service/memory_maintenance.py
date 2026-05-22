from __future__ import annotations

import logging

from .memory_conflict_resolver import MemoryConflictResolver
from .memory_consolidator import MemoryConsolidator
from .memory_store import MemoryStore
from .models import (
    MemoryConflictResolutionRequest,
    MemoryConsolidationRequest,
    MemoryDecayRequest,
    MemoryMaintenanceEvaluation,
    MemoryMaintenanceRequest,
    MemoryMaintenanceResponse,
    MemoryStats,
)

logger = logging.getLogger("memory_maintenance_service")


class MemoryMaintenanceService:
    def __init__(
        self,
        store: MemoryStore,
        conflict_resolver: MemoryConflictResolver,
        consolidator: MemoryConsolidator,
    ) -> None:
        self.store = store
        self.conflict_resolver = conflict_resolver
        self.consolidator = consolidator

    def stats(self) -> MemoryStats:
        records = self.store.list_records(include_archived=True)
        total_count = len(records)
        archived_count = sum(1 for record in records if record.archived)
        pending_conflict_count = sum(
            1
            for record in records
            if str(record.metadata.get("resolution_status", "")).lower() == "pending"
        )
        consolidated_count = sum(
            1 for record in records if "consolidated" in {tag.lower() for tag in record.tags}
        )
        return MemoryStats(
            total_count=total_count,
            active_count=total_count - archived_count,
            archived_count=archived_count,
            pending_conflict_count=pending_conflict_count,
            consolidated_count=consolidated_count,
        )

    def maintain(self, request: MemoryMaintenanceRequest) -> MemoryMaintenanceResponse:
        stats_before = self.stats()
        conflict_resolution = None
        consolidation = None
        decay = None

        if request.resolve_conflicts:
            conflict_resolution = self.conflict_resolver.resolve(
                MemoryConflictResolutionRequest(
                    include_archived=request.include_archived,
                    resolution_strategy=request.resolution_strategy,
                    persist_resolution=True,
                )
            )

        if request.consolidate_memories:
            consolidation = self.consolidator.consolidate(
                MemoryConsolidationRequest(
                    include_archived=request.include_archived,
                    min_group_size=request.min_consolidation_group_size,
                    persist_consolidated=True,
                    archive_sources=request.archive_consolidated_sources,
                )
            )

        if request.decay_memories:
            decay = self.store.decay(
                MemoryDecayRequest(
                    threshold=request.decay_threshold,
                    max_idle_days=request.max_idle_days,
                    include_archived=request.include_archived,
                )
            )

        stats_after = self.stats()
        response = MemoryMaintenanceResponse(
            stats_before=stats_before,
            stats_after=stats_after,
            conflict_resolution=conflict_resolution,
            consolidation=consolidation,
            decay=decay,
            evaluation=MemoryMaintenanceEvaluation(
                ran_conflict_resolution=request.resolve_conflicts,
                ran_consolidation=request.consolidate_memories,
                ran_decay=request.decay_memories,
            ),
        )
        logger.info(
            "memory_maintenance_completed",
            extra={
                "stats_before": stats_before.model_dump(mode="json"),
                "stats_after": stats_after.model_dump(mode="json"),
            },
        )
        return response
