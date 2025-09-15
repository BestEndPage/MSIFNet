import yaml

def getConfigYaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        try:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
            return config_dict
        except ValueError:
            print('INVALID YAML file format.. Please provide a good yaml file')
            exit(-1)

if __name__ == "__main__":
    a= getConfigYaml("./train_256.yaml")
    sys_state = {}
    for item in a.items():
        sys_state[item[0]] = item[1]