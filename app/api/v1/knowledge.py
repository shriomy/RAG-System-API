"""/knowledge — upload, list, index and delete knowledge files."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Query, UploadFile, status

from app.api.deps import CurrentUser, KnowledgeServiceDep, SettingsDep
from app.core.errors import PayloadTooLargeError, ValidationError
from app.schemas.common import MessageResponse
from app.schemas.knowledge import (
    IndexPendingResponse,
    KnowledgeFileResponse,
    KnowledgeStatsResponse,
    SignedUrlResponse,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get(
    "",
    response_model=list[KnowledgeFileResponse],
    summary="List an assistant's knowledge files",
)
async def list_files(
    user: CurrentUser,
    service: KnowledgeServiceDep,
    assistant_id: str = Query(description="Assistant whose files to list"),
) -> list[KnowledgeFileResponse]:
    files = await service.list_files(assistant_id, user.id)
    return [KnowledgeFileResponse.of(f) for f in files]


@router.post(
    "/upload",
    response_model=KnowledgeFileResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF/TXT/MD file and index it",
    description=(
        "Stores the file, registers it with `status='processing'`, and returns "
        "immediately. Indexing (load -> split -> embed -> upsert) runs in the "
        "background; poll this endpoint's list view until `status` becomes "
        "`indexed` or `failed`."
    ),
)
async def upload_file(
    user: CurrentUser,
    service: KnowledgeServiceDep,
    settings: SettingsDep,
    background: BackgroundTasks,
    assistant_id: str = Query(description="Assistant to attach this file to"),
    file: UploadFile = File(description="PDF, TXT or Markdown file"),
) -> KnowledgeFileResponse:
    if not file.filename:
        raise ValidationError("The upload has no filename.")

    # Guard on the declared size before reading, so an oversized upload is not
    # pulled fully into memory first.
    if file.size is not None and file.size > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"File exceeds the {settings.max_upload_mb} MB limit.",
            details={"size_bytes": file.size},
        )

    content = await file.read()

    record = await service.ingest_upload(
        user_id=user.id,
        assistant_id=assistant_id,
        filename=file.filename,
        content=content,
    )

    background.add_task(service.index_file, record.id, user.id)
    return KnowledgeFileResponse.of(record)


@router.post(
    "/index-pending",
    response_model=IndexPendingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Index files uploaded directly to Supabase Storage",
    description=(
        "The frontend can upload straight to Supabase Storage and insert the "
        "`knowledge_files` row itself. Those rows sit at `status='processing'` "
        "with nothing embedded. Call this after such an upload to run the "
        "pipeline over every pending file for the assistant. Also serves as the "
        "recovery path if the API restarts mid-ingestion."
    ),
)
async def index_pending(
    user: CurrentUser,
    service: KnowledgeServiceDep,
    background: BackgroundTasks,
    assistant_id: str = Query(description="Assistant whose pending files to index"),
) -> IndexPendingResponse:
    files = await service.list_files(assistant_id, user.id)
    pending = [f for f in files if str(f.status) == "processing"]

    background.add_task(service.index_pending_for_assistant, assistant_id, user.id)
    return IndexPendingResponse(scheduled=len(pending), assistant_id=assistant_id)


@router.get(
    "/stats",
    response_model=KnowledgeStatsResponse,
    summary="Knowledge-base statistics for an assistant",
)
async def stats(
    user: CurrentUser,
    service: KnowledgeServiceDep,
    assistant_id: str = Query(description="Assistant to report on"),
) -> KnowledgeStatsResponse:
    return KnowledgeStatsResponse.model_validate(await service.stats(assistant_id, user.id))


@router.get(
    "/{file_id}", response_model=KnowledgeFileResponse, summary="Get one knowledge file"
)
async def get_file(
    file_id: str, user: CurrentUser, service: KnowledgeServiceDep
) -> KnowledgeFileResponse:
    return KnowledgeFileResponse.of(await service.get_file(file_id, user.id))


@router.get(
    "/{file_id}/download-url",
    response_model=SignedUrlResponse,
    summary="Signed URL for the original file",
)
async def download_url(
    file_id: str,
    user: CurrentUser,
    service: KnowledgeServiceDep,
    expires_in: int = Query(default=3600, ge=60, le=604_800),
) -> SignedUrlResponse:
    url = await service.signed_url(file_id, user.id, expires_in)
    return SignedUrlResponse(url=url, expires_in=expires_in)


@router.post(
    "/{file_id}/reindex",
    response_model=KnowledgeFileResponse,
    summary="Re-run indexing for one file",
    description=(
        "Runs synchronously and returns the file's final state, so it is the "
        "easiest way to see why an ingestion failed — check `error_message`."
    ),
)
async def reindex_file(
    file_id: str, user: CurrentUser, service: KnowledgeServiceDep
) -> KnowledgeFileResponse:
    return KnowledgeFileResponse.of(await service.reindex_file(file_id, user.id))


@router.delete(
    "/{file_id}",
    response_model=MessageResponse,
    summary="Delete a file, its blob and its vectors",
)
async def delete_file(
    file_id: str, user: CurrentUser, service: KnowledgeServiceDep
) -> MessageResponse:
    await service.delete_file(file_id, user.id)
    return MessageResponse(message="Knowledge file deleted.")
