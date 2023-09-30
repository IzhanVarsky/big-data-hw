from ansible_vault import Vault


def read_credentials_from_file(filepath, ansible_password):
    vault = Vault(ansible_password)
    with open(filepath) as vault_data:
        return vault.load(vault_data.read())


def write_credentials_to_file(data, filepath, ansible_password):
    vault = Vault(ansible_password)
    vault.dump(data, open(filepath, 'w'))
