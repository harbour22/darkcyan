from __future__ import print_function

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from rich.progress import Progress

from darkcyan.constants import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_GOOGLEDRIVE_SCOPE,
    DEFAULT_TRAINING_YOLO_CONFIG_DIR,
)


def get_credentials():
    creds = None
    token_file = DEFAULT_CONFIG_DIR / "token.json"
    credentials_file = DEFAULT_CONFIG_DIR / "credentials.json"

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(
            token_file, DEFAULT_GOOGLEDRIVE_SCOPE
        )
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, DEFAULT_GOOGLEDRIVE_SCOPE
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_file, "w") as token:
            token.write(creds.to_json())
    return creds


def get_file_id(file_name, parent_directory_id=None, is_root=False, is_directory=False):
    if not is_root and parent_directory_id is None:
        raise ValueError("parent_directory_id must be provided if is_root is False")

    query = f"mimeType {'=' if is_directory else '!='} 'application/vnd.google-apps.folder' and '{'root' if is_root else parent_directory_id}' in parents and name = '{file_name}' and trashed = false"

    creds = get_credentials()
    try:
        service = build("drive", "v3", credentials=creds)

        # Call the Drive v3 API
        results = (
            service.files()
            .list(q=query, pageSize=10, fields="nextPageToken, files(id, name)")
            .execute()
        )
        items = results.get("files", [])

        if not items:
            print(
                f"Unable to find {file_name} on the drive for parent_directory_id {parent_directory_id}"
            )
            return []

        return items

    except HttpError as error:
        raise error


def delete_file(file_id):
    creds = get_credentials()
    try:
        service = build("drive", "v3", credentials=creds)

        # Call the Drive v3 API
        service.files().delete(fileId=file_id).execute()
        print(f"{file_id} deleted")

    except HttpError as error:
        raise error


def get_directory_id_from_path(parent_directory_path):
    creds = get_credentials()

    parent_directory_id = None
    for location in parent_directory_path.split("/"):
        files = get_file_id(
            location,
            is_root=True if parent_directory_path.index(location) == 0 else False,
            parent_directory_id=parent_directory_id,
            is_directory=True,
        )
        if len(files) > 1:
            raise ValueError(
                f"Multiple directories found with name {location} in parent_directory_id {parent_directory_id}"
            )
        parent_directory_name = files[0]["name"]
        parent_directory_id = files[0]["id"]
        print(
            f"parent_directory_name: {parent_directory_name}, parent_directory_id: {parent_directory_id}"
        )
    return parent_directory_id


def upload_file(upload_file_path, parent_id, mimetype="application/zip"):
    creds = get_credentials()
    with Progress(transient=True) as progress:
        task1 = progress.add_task(
            f"[blue]Uploading {upload_file_path.name} to google drive", total=None
        )

        try:
            # create drive api client
            service = build("drive", "v3", credentials=creds)

            file_metadata = {"name": upload_file_path.name, "parents": [parent_id]}
            media = MediaFileUpload(upload_file_path, mimetype=mimetype, resumable=True)
            # pylint: disable=maybe-no-member

            file = (
                service.files()
                .create(body=file_metadata, media_body=media, fields="id")
                .execute()
            )
            progress.update(task1, completed=1)

        except HttpError as error:
            print(f"An error occurred: {error}")
            file = None
            raise error

        return file.get("id")


if __name__ == "__main__":
    parent_directory_id = get_directory_id_from_path(
        f"{DEFAULT_TRAINING_YOLO_CONFIG_DIR}"
    )
    files = get_file_id("limetree_yolo_training_config.json", parent_directory_id)
    print(files)
    for file in files:
        delete_file(file["id"])

    # temp_dir = Path(Config.get_value('temp_dir'))
    # zip_filename = temp_dir / f"{Config.get_value('data_prefix')}_v{version}_classify.zip"
    # version = 4.1
    # upload_file(zip_filename, parent_directory_id)
