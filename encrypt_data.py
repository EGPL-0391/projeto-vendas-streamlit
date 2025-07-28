"""
Script para criptografar o arquivo Excel com dados sensíveis
Execute este script LOCALMENTE para gerar os dados criptografados
"""

import base64
from cryptography.fernet import Fernet
import os

def generate_key():
    """Gera uma nova chave de criptografia"""
    return Fernet.generate_key()

def encrypt_file(file_path, key):
    """Criptografa um arquivo e retorna em base64"""
    with open(file_path, 'rb') as file:
        file_data = file.read()
    
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(file_data)
    
    # Converte para base64 para armazenar como string
    encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
    
    return encrypted_b64

def main():
    # Caminho para seu arquivo Excel
    excel_path = r'C:\Users\EdvaldoGuilhermePaiv\Documents\GitHub\projeto-vendas-streamlit\data\base_vendas_24.xlsx'
    
    if not os.path.exists(excel_path):
        print(f"❌ Arquivo não encontrado: {excel_path}")
        print("Certifique-se de que o caminho está correto.")
        return
    
    # Gera chave de criptografia
    key = generate_key()
    print("🔑 Chave de criptografia gerada:")
    print(key.decode('utf-8'))
    print()
    
    # Criptografa o arquivo
    try:
        encrypted_data = encrypt_file(excel_path, key)
        print("✅ Arquivo criptografado com sucesso!")
        print()
        print("📋 Dados criptografados (copie para o secrets.toml):")
        print(f'encrypted_file = """{encrypted_data}"""')
        print()
        
        # Salva em arquivo local para referência
        with open("encrypted_data.txt", "w") as f:
            f.write(f"# Chave de criptografia:\n")
            f.write(f"key = \"{key.decode('utf-8')}\"\n\n")
            f.write(f"# Dados criptografados:\n")
            f.write(f"encrypted_file = \"\"\"{encrypted_data}\"\"\"\n")
        
        print("💾 Dados salvos em 'encrypted_data.txt' para referência")
        print()
        print("⚠️  IMPORTANTE:")
        print("1. Copie a chave e os dados para o arquivo secrets.toml")
        print("2. NÃO commite o arquivo encrypted_data.txt no GitHub")
        print("3. Mantenha a chave em local seguro")
        
    except Exception as e:
        print(f"❌ Erro ao criptografar: {str(e)}")

if __name__ == "__main__":
    main()