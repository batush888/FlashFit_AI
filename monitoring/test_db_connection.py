#!/usr/bin/env python3
"""
Database Connection Test Script for FlashFit AI Monitoring
This script tests PostgreSQL and Redis connectivity with the same configuration
used by the monitoring system.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_postgresql():
    """Test PostgreSQL connection"""
    try:
        import psycopg2
        
        # Get database connection parameters from environment
        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'flashfit_ai'),
            'user': os.getenv('POSTGRES_USER', 'flashfit_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'flashfit_dev_password')
        }
        
        print(f"Testing PostgreSQL connection to {db_config['host']}:{db_config['port']}/{db_config['database']}...")
        
        # Test connection
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        cursor = conn.cursor()
        cursor.execute('SELECT version()')
        version = cursor.fetchone()
        version_str = version[0] if version and len(version) > 0 else 'unknown'
        print(f"✓ PostgreSQL connection successful: {version_str}")
        
        # Test if monitoring tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('system_metrics', 'query_metrics', 'health_checks')
        """)
        tables = cursor.fetchall()
        print(f"✓ Found {len(tables)} monitoring tables: {[t[0] for t in tables]}")
        
        cursor.close()
        conn.close()
        return True
        
    except ImportError:
        print("✗ psycopg2 not installed. Install with: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")
        return False

def test_redis():
    """Test Redis connection"""
    try:
        import redis
        
        # Get Redis connection parameters from environment
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        
        print(f"Testing Redis connection to {redis_host}:{redis_port}...")
        
        # Test connection
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        r.ping()
        
        # Get Redis info
        info = r.info()
        redis_version = info.get('redis_version', 'unknown') if info else 'unknown'
        print(f"✓ Redis connection successful: {redis_version}")
        
        # Test basic operations
        r.set('test_key', 'test_value', ex=10)
        value = r.get('test_key')
        print(f"✓ Redis operations working: {value if value else 'None'}")
        
        return True
        
    except ImportError:
        print("✗ redis not installed. Install with: pip install redis")
        return False
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False

def main():
    """Main test function"""
    print("FlashFit AI Database Connection Test")
    print("=" * 40)
    
    # Test PostgreSQL
    postgres_ok = test_postgresql()
    print()
    
    # Test Redis
    redis_ok = test_redis()
    print()
    
    # Summary
    print("Summary:")
    print(f"PostgreSQL: {'✓ OK' if postgres_ok else '✗ FAILED'}")
    print(f"Redis: {'✓ OK' if redis_ok else '✗ FAILED'}")
    
    if postgres_ok and redis_ok:
        print("\n🎉 All database connections are working!")
        sys.exit(0)
    else:
        print("\n❌ Some database connections failed. Check configuration.")
        sys.exit(1)

if __name__ == '__main__':
    main()