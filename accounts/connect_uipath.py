## This script requires "requests": http://docs.python-requests.org/
## To install: pip install requests

import requests
import json

data = {"grant_type": "refresh_token",
        "client_id": "8DEv1AMNXczW3y4U15LL3jYf62jK93n5",
        "refresh_token": "1vIR8Z4Y4UYYAJelrHgI8fqXyLlW8rW-HYh855FlWgJ1K"}

headers = {"Content-Type": "application/json",
           "X-UIPATH-TenantName": "DefaultTenant"}

r = requests.post("https://account.uipath.com/oauth/token", data, headers)

response = json.loads(r.content)
auth = "Bearer " + response["access_token"]

headers2 = {"Content-Type": "application/json",
            "X-UIPATH-TenantName": "DefaultTenant",
           "X-UIPATH-OrganizationUnitId":"921424",
            "Authorization": auth}

## Process without parameters (Simple)
startInfo = {}
startInfo['ReleaseKey'] = '4a6deed1-9b68-4397-805c-8af0ac7d115d'
startInfo['Strategy'] = 'All'

"""
## Process with parameters
startInfo = {}
startInfo['ReleaseKey'] = '921424'
startInfo['Strategy'] = 'All'
startInfo['InputArguments'] = '{\"param1\":\"Test from Python\",\"param2\":\"Video Live 20:42\"}'
"""

data2 = {}
data2['startInfo'] = startInfo
json_data = json.dumps(data2)

r2 = requests.post(
    "https://platform.uipath.com/nedrfotjyo/DefaultTenant/odata/Jobs/UiPath.Server.Configuration.OData.StartJobs",
    data=json_data, headers=headers2)
print(r2.content)

