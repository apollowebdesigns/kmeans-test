import re
import json

subbed = re.sub(r'^[^,]', r'"', str({'hello': 'dear'}))
subbed = re.sub(r'\'', r'"', str({'hello': 'dear'}))
subbed = re.sub(r'None', r'null', str({'hello': None}))
print(subbed)

print(json.loads(subbed))