"""Helper script to diagnose and fix .env file issues."""

import os
import re

def diagnose_env():
    """Diagnose .env file issues."""
    env_path = ".env"
    
    if not os.path.exists(env_path):
        print("❌ .env file not found!")
        return False
    
    print("🔍 Diagnosing .env file...\n")
    
    with open(env_path, "r") as f:
        lines = f.readlines()
    
    issues = []
    has_supabase = False
    
    for i, line in enumerate(lines, 1):
        line = line.rstrip()
        
        # Skip empty lines and comments
        if not line or line.strip().startswith("#"):
            continue
        
        # Check for common issues
        if " = " in line:
            issues.append(f"Line {i}: Has spaces around = (should be KEY=value)")
        elif ":" in line and "=" not in line:
            issues.append(f"Line {i}: Using : instead of = (should be KEY=value)")
        elif line.count("=") == 0:
            issues.append(f"Line {i}: Missing = sign")
        elif line.count("=") > 1:
            issues.append(f"Line {i}: Multiple = signs (may need quotes)")
        
        # Check for Supabase
        if line.startswith("SUPABASE_URL"):
            has_supabase = True
            if "your" in line.lower() or "xxx" in line.lower():
                issues.append(f"Line {i}: SUPABASE_URL has placeholder value")
        if line.startswith("SUPABASE_KEY"):
            if "your" in line.lower() or "xxx" in line.lower():
                issues.append(f"Line {i}: SUPABASE_KEY has placeholder value")
    
    if issues:
        print("⚠️  Found issues:\n")
        for issue in issues:
            print(f"   {issue}")
        print()
    else:
        print("✅ No formatting issues found\n")
    
    # Check for required keys
    print("📋 Checking required configuration:\n")
    
    required = {
        "ANTHROPIC_API_KEY": False,
        "TAVILY_API_KEY": False,
        "SUPABASE_URL": False,
        "SUPABASE_KEY": False,
    }
    
    with open(env_path, "r") as f:
        content = f.read()
        for key in required:
            if key in content:
                # Check if it's not a placeholder
                pattern = rf"{key}=([^\n]+)"
                match = re.search(pattern, content)
                if match:
                    value = match.group(1).strip()
                    if value and "your" not in value.lower() and "xxx" not in value.lower():
                        required[key] = True
                        print(f"   ✅ {key} is set")
                    else:
                        print(f"   ⚠️  {key} has placeholder value")
                else:
                    print(f"   ❌ {key} not found")
            else:
                print(f"   ❌ {key} not found")
    
    print()
    
    # Summary
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"❌ Missing or incomplete: {', '.join(missing)}")
        return False
    else:
        print("✅ All required configuration present!")
        return True

def add_supabase_interactive():
    """Interactively add Supabase credentials."""
    env_path = ".env"
    
    print("\n📝 Adding Supabase credentials...\n")
    print("Get your credentials from: https://supabase.com/dashboard → Settings → API\n")
    
    url = input("SUPABASE_URL (e.g., https://xxx.supabase.co): ").strip()
    if not url:
        print("❌ URL required")
        return False
    
    key = input("SUPABASE_KEY (Service Role Key): ").strip()
    if not key:
        print("❌ Key required")
        return False
    
    # Read current .env
    with open(env_path, "r") as f:
        content = f.read()
    
    # Remove old Supabase entries if any
    lines = content.split("\n")
    lines = [l for l in lines if not l.startswith("SUPABASE_") and not l.startswith("ENABLE_CACHING") and not l.startswith("CACHE_TTL")]
    
    # Add new entries
    supabase_config = f"""
# Supabase Configuration
SUPABASE_URL={url}
SUPABASE_KEY={key}
ENABLE_CACHING=true
CACHE_TTL_HOURS=24
"""
    
    # Write back
    with open(env_path, "w") as f:
        f.write("\n".join(lines).strip())
        f.write(supabase_config)
    
    print("\n✅ Supabase credentials added to .env")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("The Oracle - .env File Diagnostic")
    print("=" * 50)
    print()
    
    if diagnose_env():
        print("✅ .env file looks good!")
    else:
        print("\n" + "=" * 50)
        response = input("\nAdd Supabase credentials now? [y/N]: ")
        if response.lower() == "y":
            add_supabase_interactive()
            print("\n🔍 Re-checking...")
            diagnose_env()
