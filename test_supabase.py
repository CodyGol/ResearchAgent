"""Test script to verify Supabase connection and integration."""

import asyncio
import sys

from config import settings


async def test_supabase_connection():
    """Test Supabase client connection."""
    print("🔍 Testing Supabase Connection...\n")

    # Test 1: Configuration
    print("1️⃣ Checking configuration...")
    try:
        if not settings.supabase_url:
            print("   ❌ SUPABASE_URL not set in .env")
            return False
        if not settings.supabase_key:
            print("   ❌ SUPABASE_KEY not set in .env")
            return False
        print(f"   ✅ URL: {settings.supabase_url[:40]}...")
        print(f"   ✅ Key: {settings.supabase_key[:20]}...")
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return False

    # Test 2: Client initialization
    print("\n2️⃣ Testing client initialization...")
    try:
        from db.client import get_supabase_client

        client = get_supabase_client()
        print("   ✅ Supabase client created")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Run: pip install supabase postgrest")
        return False
    except Exception as e:
        print(f"   ❌ Client error: {e}")
        return False

    # Test 3: Database connection
    print("\n3️⃣ Testing database connection...")
    try:
        # Try a simple query to test connection
        response = client.table("research_plans").select("id").limit(1).execute()
        print("   ✅ Database connection successful")
        print(f"   ✅ Table 'research_plans' exists")
    except Exception as e:
        error_str = str(e).lower()
        if "does not exist" in error_str or "relation" in error_str:
            print("   ⚠️  Table 'research_plans' not found")
            print("   💡 Run the SQL from db/schema.sql in Supabase SQL Editor")
            return False
        else:
            print(f"   ❌ Connection error: {e}")
            return False

    # Test 4: Repository operations
    print("\n4️⃣ Testing repository operations...")
    try:
        from db.repository import _get_plan_repo

        # Get repository (lazy initialization)
        plan_repo = _get_plan_repo()

        # Test cache operations (should work even if empty)
        test_query = "test query for connection verification"
        cached = await plan_repo.get_cached_plan(test_query)
        print("   ✅ Repository operations working")
        if cached:
            print("   ℹ️  Found cached plan (unexpected for test query)")
        else:
            print("   ℹ️  No cached plan found (expected)")
    except Exception as e:
        print(f"   ❌ Repository error: {e}")
        return False

    # Test 5: Check all tables
    print("\n5️⃣ Checking required tables...")
    tables = ["research_plans", "research_reports", "search_results"]
    missing = []
    for table in tables:
        try:
            client.table(table).select("id").limit(1).execute()
            print(f"   ✅ Table '{table}' exists")
        except Exception:
            print(f"   ❌ Table '{table}' missing")
            missing.append(table)

    if missing:
        print(f"\n   ⚠️  Missing tables: {', '.join(missing)}")
        print("   💡 Run the SQL from db/schema.sql in Supabase SQL Editor")
        return False

    print("\n" + "=" * 50)
    print("✅ All Supabase tests passed!")
    print("=" * 50)
    return True


async def test_full_system():
    """Test the full system with a simple query."""
    print("\n🚀 Testing Full System Integration...\n")

    try:
        from graph import create_graph

        graph = create_graph()
        app = graph.compile()

        initial_state = {
            "user_query": "What is artificial intelligence?",
            "research_plan": None,
            "research_results": None,
            "critique": None,
            "final_report": None,
            "current_node": "planner",
            "iteration_count": 0,
            "error": None,
        }

        print("📝 Query: What is artificial intelligence?")
        print("⏳ Processing...\n")

        final_state = await app.ainvoke(initial_state)

        if final_state.get("error"):
            print(f"❌ Error: {final_state['error']}")
            return False

        report = final_state.get("final_report")
        if report:
            print("✅ Research Complete!")
            print(f"📊 Report generated ({len(report.content)} chars)")
            print(f"📚 Sources: {len(report.sources)}")
            print(f"🎯 Confidence: {report.confidence:.2f}")

            # Check if saved to database
            if settings.enable_caching:
                print("\n💾 Checking if report was saved to Supabase...")
                try:
                    from db.repository import _get_report_repo

                    report_repo = _get_report_repo()
                    reports = await report_repo.list_reports(limit=1)
                    if reports:
                        print(f"   ✅ Latest report found in database (ID: {reports[0].id})")
                    else:
                        print("   ⚠️  No reports found (may need to check)")
                except Exception as e:
                    print(f"   ⚠️  Could not verify save: {e}")

            return True
        else:
            print("⚠️  No report generated")
            return False

    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("=" * 50)
    print("The Oracle - Supabase Integration Test")
    print("=" * 50)

    # Test Supabase connection
    supabase_ok = await test_supabase_connection()

    if not supabase_ok:
        print("\n❌ Supabase tests failed. Please fix the issues above.")
        sys.exit(1)

    # Test full system (optional, can be slow)
    print("\n" + "=" * 50)
    response = input("Run full system test? (This will make API calls) [y/N]: ")
    if response.lower() == "y":
        system_ok = await test_full_system()
        if system_ok:
            print("\n✅ All tests passed!")
        else:
            print("\n⚠️  System test had issues")
    else:
        print("\n✅ Supabase connection verified!")


if __name__ == "__main__":
    asyncio.run(main())
