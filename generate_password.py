"""
Script para gerar hashes seguros de senhas para usuários
Execute este script LOCALMENTE para gerar as credenciais
"""

import hashlib
import base64
import secrets
import string

def generate_salt():
    """Gera um salt aleatório"""
    return secrets.token_hex(32)

def hash_password(password, salt):
    """Gera hash seguro da senha com salt"""
    password_hash = hashlib.pbkdf2_hmac('sha256', 
                                      password.encode('utf-8'), 
                                      salt.encode('utf-8'), 
                                      100000)  # 100.000 iterações
    return base64.b64encode(password_hash).decode('ascii')

def generate_random_password(length=12):
    """Gera uma senha aleatória forte"""
    characters = string.ascii_letters + string.digits + "!@#$%&*"
    return ''.join(secrets.choice(characters) for _ in range(length))

def create_user(username, password=None):
    """Cria um usuário com senha segura"""
    if password is None:
        password = generate_random_password()
        print(f"🔐 Senha gerada automaticamente para {username}: {password}")
    
    salt = generate_salt()
    password_hash = hash_password(password, salt)
    
    return {
        "username": username,
        "password": password,
        "salt": salt,
        "password_hash": password_hash
    }

def main():
    print("🔐 GERADOR DE CREDENCIAIS SEGURAS")
    print("=" * 50)
    
    users = []
    
    while True:
        print()
        username = input("👤 Digite o nome do usuário (ou ENTER para finalizar): ").strip()
        
        if not username:
            break
            
        print("Escolha uma opção:")
        print("1. Digitar senha manualmente")
        print("2. Gerar senha automaticamente")
        
        choice = input("Opção (1 ou 2): ").strip()
        
        if choice == "1":
            password = input("🔑 Digite a senha: ").strip()
            if not password:
                print("❌ Senha não pode estar vazia!")
                continue
        else:
            password = None
        
        user_data = create_user(username, password)
        users.append(user_data)
        
        print(f"✅ Usuário '{username}' criado com sucesso!")
    
    if not users:
        print("❌ Nenhum usuário criado.")
        return
    
    print()
    print("📋 CONFIGURAÇÃO PARA secrets.toml:")
    print("=" * 50)
    print()
    print("[users]")
    
    for user in users:
        print(f'[users.{user["username"]}]')
        print(f'password_hash = "{user["password_hash"]}"')
        print(f'salt = "{user["salt"]}"')
        print()
    
    # Salva em arquivo para referência
    with open("user_credentials.txt", "w") as f:
        f.write("# CREDENCIAIS DOS USUÁRIOS\n")
        f.write("# MANTENHA ESTE ARQUIVO SEGURO!\n\n")
        
        for user in users:
            f.write(f"Usuário: {user['username']}\n")
            f.write(f"Senha: {user['password']}\n")
            f.write(f"Salt: {user['salt']}\n")
            f.write(f"Hash: {user['password_hash']}\n")
            f.write("-" * 50 + "\n")
        
        f.write("\n# CONFIGURAÇÃO PARA secrets.toml:\n")
        f.write("[users]\n")
        for user in users:
            f.write(f'[users.{user["username"]}]\n')
            f.write(f'password_hash = "{user["password_hash"]}"\n')
            f.write(f'salt = "{user["salt"]}"\n\n')
    
    print("💾 Credenciais salvas em 'user_credentials.txt'")
    print()
    print("⚠️  IMPORTANTE:")
    print("1. Copie a configuração para o arquivo secrets.toml")
    print("2. NÃO commite o arquivo user_credentials.txt no GitHub")
    print("3. Compartilhe as senhas com os usuários de forma segura")
    print("4. Mantenha backup das credenciais em local seguro")

if __name__ == "__main__":
    main()