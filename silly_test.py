import yaml

try: 
    with open ("conf/base/catalog.yml", 'r') as file:
        config = yaml.safe_load(file)
        print(config)
except Exception as e:
    print('Error reading the config file')