from enum import Enum
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod

from .utils import get_firebase_client, tick
from .logger import get_logger, StructuredLogger
from .cloudtasks.queue import QueueType

logger = get_logger(__name__)
struct_logger = StructuredLogger(__name__)


# Collection name
RESULT_COLLECTION_NAME = "TaskRecords"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRecord(BaseModel):
    taskId: str
    queueType: QueueType
    status: TaskStatus
    createdAt: int
    updatedAt: int
    endpoint: Optional[str] = None
    payload: Optional[str] = None
    resultUrl: Optional[str] = None
    error: Optional[str] = None


class AbstractPollManager(ABC):
    """Abstract base class for storing task results"""

    @abstractmethod
    def create_task_record(
        self,
        task_id: str,
        queue_type: QueueType,
        status: TaskStatus,
        endpoint: Optional[str] = None,
        payload: Optional[str] = None,
        result_url: Optional[str] = None,
        error: Optional[str] = None,
    ):
        pass

    @abstractmethod
    def update_task_record(
        self,
        task_id: str,
        status: TaskStatus,
        endpoint: Optional[str] = None,
        payload: Optional[str] = None,
        result_url: Optional[str] = None,
        error: Optional[str] = None,
    ):
        pass

    @abstractmethod
    def get_task_record(self, task_id: str) -> Optional[TaskRecord]:
        pass


class PollManager(AbstractPollManager):
    """Firestore implementation for storing task results

    Args:
        collection_name: Firebase Document to save to
    """

    def __init__(self, collection_name: str = RESULT_COLLECTION_NAME):
        super().__init__()

        # Initialize the Firebase client
        self.db = get_firebase_client()
        self.collection = collection_name

    def create_task_record(
        self,
        task_id: str,
        queue_type: QueueType,
        status: TaskStatus,
        endpoint: Optional[str] = None,
        payload: Optional[str] = None,
        result_url: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Create a new task result document in Firestore.

        Args:
            queue_type: Cloud Task queue type/name
            task_id: ID for the Cloud Task
            status: Initial task status (default: PENDING)
            endpoint: Endpoint associated with this task
            payload: Jsonified payload as a string
            result: Optional result data
            error: Optional error message)

        Return:
        TaskRecord
        """
        doc_ref = self.db.collection(self.collection).document(task_id)

        # Check if document already exists
        doc = doc_ref.get()
        if doc.exists:
            logger.error(f"create_task_result - Task {task_id} already exists")
            raise ValueError(f"create_task_result - Task {task_id} already exists")

        # Get current timestamp
        now = tick()

        # Prepare document data
        document_data = {
            "taskId": task_id,
            "queueType": queue_type.value,
            "status": status.value,
            "createdAt": now,
            "updatedAt": now,
        }

        # Add optional fields
        if result_url is not None:
            document_data["resultUrl"] = result_url
        if endpoint is not None:
            document_data["endpoint"] = endpoint
        if payload is not None:
            document_data["payload"] = payload
        if error is not None:
            document_data["error"] = error

        # Create the document
        doc_ref.set(document_data)

        struct_logger.info(
            f"create_task_result - Created new {RESULT_COLLECTION_NAME} {task_id}",
            **document_data,
        )

    def update_task_record(
        self,
        task_id: str,
        status: TaskStatus,
        endpoint: Optional[str] = None,
        payload: Optional[str] = None,
        result_url: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Set task status in Firestore.

        Args:
            task_id: ID for the Cloud Task
            status: Task status
        """
        doc_ref = self.db.collection(self.collection).document(task_id)

        # Check if document exists
        doc = doc_ref.get()

        if not doc.exists:
            logger.error(f"update_task_result - Task {task_id} not found")
            raise ValueError(f"update_task_result - Task {task_id} not found")

        # Get current timestamp
        now = tick()

        # Prepare update data
        update_data: Dict[str, Any] = {
            "status": status.value,
            "updatedAt": now,
        }

        # Add optional fields
        if result_url is not None:
            update_data["resultUrl"] = result_url
        if endpoint is not None:
            update_data["endpoint"] = endpoint
        if payload is not None:
            update_data["payload"] = payload
        if error is not None:
            update_data["error"] = error

        struct_logger.info(
            f"update_task_result - Updating {RESULT_COLLECTION_NAME} {doc.id}",
            **update_data,
        )
        # Update existing document
        doc_ref.update(update_data)

    def get_task_record(self, task_id: str) -> Optional[TaskRecord]:
        """Get task record from Firestore

        Args:
            task_id: Cloud Task ID

        Return:
        TaskRecord
        """
        doc_ref = self.db.collection(self.collection).document(task_id)
        doc = doc_ref.get()

        if not doc.exists:
            return None

        data = doc.to_dict()
        if data is None:
            return None

        return TaskRecord(
            taskId=task_id,
            queueType=QueueType(data["queueType"]),
            status=TaskStatus(data["status"]),
            createdAt=data.get("createdAt", 0),
            updatedAt=data.get("updatedAt", 0),
            endpoint=data.get("endpoint"),
            payload=data.get("payload"),
            resultUrl=data.get("resultUrl"),
            error=data.get("error"),
        )

    def get_task_records_from_queue(
        self,
        queue_type: QueueType,
        page: int = 0,
        limit: int = 100,
    ) -> List[TaskRecord]:
        """Fetch task records from the queue.

        Args:
            queue_type:
            limit: Maximum number to fetch

        Return:
            List of task records
        """
        try:
            # Query Firestore for tasks matching the queue type
            query = self.db.collection(self.collection)
            query = query.where("queueType", "==", queue_type.value)
            query = query.order_by("createdAt", direction="DESCENDING")
            query = query.offset(page * limit)
            query = query.limit(limit)

            # Execute query and convert to TaskRecord objects
            task_records = []
            for doc in query.stream():
                data = doc.to_dict()
                if data:
                    try:
                        task_record = TaskRecord(
                            taskId=data["taskId"],
                            queueType=QueueType(data["queueType"]),
                            status=TaskStatus(data["status"]),
                            createdAt=data.get("createdAt", 0),
                            updatedAt=data.get("updatedAt", 0),
                            endpoint=data.get("endpoint"),
                            payload=data.get("payload"),
                            resultUrl=data.get("resultUrl"),
                            error=data.get("error"),
                        )
                        task_records.append(task_record)
                    except (KeyError, ValueError) as e:
                        logger.warning(
                            f"get_task_records_from_queue - Skipping invalid task record {doc.id}: {e}"
                        )
                        continue

            logger.info(
                f"get_task_records_from_queue - Retrieved {len(task_records)} task records for queue {queue_type.value}"
            )
            return task_records

        except Exception as e:
            logger.error(
                f"get_task_records_from_queue - Failed to fetch task records for queue {queue_type.value}: {e}"
            )
            return []
