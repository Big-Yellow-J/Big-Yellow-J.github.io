import re
import json
import requests
import pandas as pd
from pathlib import Path

def authenticate(session: requests.Session, base_url: str, api_key: str) -> str:
    base = base_url.rstrip('/')
    me_url = f"{base}/api/users/me/"

    # 1) 尝试 Token header
    session.headers.update({'Authorization': f'Token {api_key}', 'Accept': 'application/json'})
    resp = session.get(me_url)
    if resp.status_code == 200:
        print('认证成功（Token 方式）')
        return api_key

    # 2) 尝试 Bearer header
    session.headers.update({'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'})
    resp = session.get(me_url)
    if resp.status_code == 200:
        print('认证成功（Bearer 方式）')
        return api_key

    # 3) Token/Bearer 都失败，尝试用 refresh token 换取 access token
    refresh_url = f"{base}/api/token/refresh/"
    resp = session.post(refresh_url, json={'refresh': api_key})
    if resp.status_code == 200:
        data = resp.json()
        access = data.get('access', '')
        if access:
            print('已通过 refresh token 换取 access token')
            return access

    raise SystemExit(
        f'认证失败: 无法使用 Token/Bearer 方式，也无法刷新 token。'
        f'请确认 API key 有效。\n刷新响应: {resp.text[:1000]}'
    )

class LabelStudioDownloader:
    def __init__(self, base_url: str, api_key: str, output_dir: str = '.', 
                 **kwargs):
        self.kwargs = kwargs
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = self.output_dir / 'labelstudio_exports'
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.columns = kwargs.get("columns", None)
        
        self._authenticated = False
        self._session = requests.Session()

    def authenticate(self) -> None:
        if self._authenticated:
            return
        try:
            access_token = authenticate(self._session, self.base_url, self.api_key)
            self._session.headers.update({
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json',
            })
            self._authenticated = True
        except requests.exceptions.RequestException as e:
            raise SystemExit(f'连接测试失败: {e}')

    @staticmethod
    def _get_json(response) -> dict | list:
        try:
            return response.json()
        except ValueError:
            raise RuntimeError(f'从 Label Studio 返回的内容无法解析为 JSON: {response.text[:200]}')

    @staticmethod
    def get_task_url(base_url: str, project_id: int, task_id: int) -> str:
        base = base_url.rstrip('/')
        return f"{base}/projects/{project_id}/data?tab={project_id}&task={task_id}"

    @staticmethod
    def extract_urls_from_data(data) -> list[str]:
        urls = []
        if isinstance(data, dict):
            for value in data.values():
                urls.extend(LabelStudioDownloader.extract_urls_from_data(value))
        elif isinstance(data, list):
            for item in data:
                urls.extend(LabelStudioDownloader.extract_urls_from_data(item))
        elif isinstance(data, str):
            if data.startswith(('http://', 'https://', 's3://')):
                urls.append(data)
            elif re.search(r'\.(jpg|jpeg|png|gif|bmp|mp4|wav|mp3|json|txt|csv)$', data, re.IGNORECASE):
                urls.append(data)
        return urls

    def _iter_api_pages(self, url: str, params: dict = None):
        """通用的 Label Studio API 分页迭代器。"""
        if params is None:
            params = {}
        while url:
            response = self._session.get(url, params=params)
            if response.status_code == 401:
                body = response.text
                raise SystemExit(
                    f"401 Unauthorized when accessing {url}. "
                    f"请确认 API key 是否正确且有权限。\n服务器响应: {body[:1000]}"
                )
            try:
                response.raise_for_status()
            except Exception as e:
                body = response.text if hasattr(response, 'text') else '<no body>'
                raise RuntimeError(f'HTTP {response.status_code} error for {url}: {body[:1000]}') from e

            payload = self._get_json(response)
            if isinstance(payload, dict) and 'results' in payload:
                items = payload['results']
                next_url = payload.get('next')
            elif isinstance(payload, dict) and 'data' in payload and isinstance(payload['data'], list):
                items = payload['data']
                next_url = payload.get('next')
            elif isinstance(payload, list):
                items = payload
                next_url = None
            else:
                raise RuntimeError(f'无法识别分页响应: {payload}')

            for item in items:
                yield item

            if not next_url:
                return
            url = next_url
            params = {}

    def get_projects(self):
        """获取所有project"""
        self.authenticate()
        #mark: 具体url 需要到具体项目中查看，可能是 /api/projects/ 也可能是 /api/projects?user=me
        url = f"{self.base_url}/api/projects/"
        yield from self._iter_api_pages(url, {'page_size': 100})

    def get_tasks(self, project_id: int):
        """获取project 中 task"""
        self.authenticate()
        #mark: 具体url 需要到具体项目中查看，可能是 /api/projects/{project_id}/tasks/ 也可能是 /api/tasks?project={project_id}
        url = f"{self.base_url}/api/projects/{project_id}/tasks/"
        yield from self._iter_api_pages(url, {'page_size': 200, 'resolve_uri': 'true'})

    def get_data(self, project: dict, task: dict, export_file: Path | None) -> dict:
        data_urls = self.extract_urls_from_data(task.get('data', {}))
        data =  {
            'project_id': project.get('id'),
            'project_title': project.get('title'),
            'project_description': project.get('description') or '',
            'project_labeled_tasks_count': project.get('tasks_count') or project.get('count_tasks') or '',
            'task_id': task.get('id'),
            'task_created_at': task.get('created_at') or '',
            'task_updated_at': task.get('updated_at') or '',
            'task_completed_by': task.get('completed_by') or '',
            'task_is_skipped': task.get('is_skipped') or task.get('skipped') or False,
            'task_annotations': json.dumps(task.get('annotations', []), ensure_ascii=False) if task.get('annotations') is not None else '',
            'task_data': json.dumps(task.get('data', {}), ensure_ascii=False),
            'data_urls': ';'.join(data_urls),
            'task_url': self.get_task_url(self.base_url, project.get('id'), task.get('id')),
            'project_export_file': str(export_file) if export_file else '',
        }
        self.columns = self.columns or data.keys()
        return data

    def export_project(self, project_id: int) -> Path:
        """下载项目的 JSON 导出文件，返回文件路径。"""
        self.authenticate()
        #mark: 具体url 需要到具体项目中查看，可能是 /api/projects/{project_id}/export/ 也可能是 /api/export?project={project_id}
        url = f"{self.base_url}/api/projects/{project_id}/export"
        params = {'exportType': 'JSON', 'download_all_tasks': 'true'}
        response = self._session.get(url, params=params, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        ext = '.json'
        if 'zip' in content_type or 'Content-Disposition' in response.headers and '.zip' in response.headers['Content-Disposition'].lower():
            ext = '.zip'
        filename = self.export_dir / f'project_{project_id}_export{ext}'
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return filename

    def run(self, csv_file: str = 'labelstudio_all_projects_tasks.csv', skip_export: bool = False,):
        """执行完整的下载和导出流程。"""
        self.authenticate()

        result = []
        for i, project in enumerate(self.get_projects()):
            pid = project.get('id')

            title = project.get('title') or str(pid)
            print(f'处理项目 {pid} - {title}')

            export_file = None
            if not skip_export:
                try:
                    export_file = self.export_project(pid)
                    print(f'  已下载导出文件: {export_file.name}')
                except Exception as exc:
                    print(f'  警告: 无法下载项目导出文件 {pid}: {exc}')

            task_count = 0
            for task in self.get_tasks(pid):
                result.append(self.get_data(project, task, export_file))
                task_count += 1
            print(f'  任务数: {task_count}')

        if not result:
            print('没有找到任何任务，CSV 不会生成。')
            return

        csv_path = self.output_dir / csv_file
        df = pd.DataFrame(result, columns= self.columns)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'CSV 已生成: {csv_path}')
        print(f'项目导出文件保存目录: {self.export_dir}')


def main():
    base_url = "http://localhost:8080"
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA4NjU3MTk0MywiaWF0IjoxNzc5MzcxOTQzLCJqdGkiOiJjOWRlZGVlYjI4NTc0ZjE2OGJjNTA4OWI2YTFkNTVjMiIsInVzZXJfaWQiOiIxIn0.z9Jm8fZ7-cntSKgQYypJq5uF5kzKDEkqopTlGIRaYmc"

    output_dir, csv_file = './label_studio_results', './label_studio.csv'
    skip_export = False

    columns = [
            'project_id', 'project_title', 
            'task_id', 
            'task_annotations', 
            'task_data', 
            'data_urls',
            'task_url', 
            'project_export_file',
        ]
    downloader = LabelStudioDownloader(
        base_url= base_url,
        api_key= api_key,
        output_dir=output_dir,
        columns= columns,
    )
    downloader.run(
        csv_file=csv_file,
        skip_export=skip_export,
    )


if __name__ == '__main__':
    main()
