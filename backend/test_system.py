import asyncio
import httpx
import os
from pathlib import Path

async def test_upload_file(client: httpx.AsyncClient, file_path: Path, expected_format: str):
    """Test file upload and processing."""
    print(f"\nTesting upload of {file_path.name}...")
    
    # Prepare file upload
    files = {'file': (file_path.name, open(file_path, 'rb'), 'application/octet-stream')}
    
    try:
        response = await client.post(
            'http://localhost:8000/api/upload',
            files=files
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"Classification: {result['classification']['format']} ({result['classification']['intent']})")
            print(f"Memory ID: {result['memory_id']}")
            print("\nTriggered Actions:")
            for action in result['actions']:
                print(f"- {action['type']}: {action['result']['status']}")
                
            # Verify format
            if result['classification']['format'] != expected_format:
                print(f"⚠️ Warning: Expected format {expected_format}, got {result['classification']['format']}")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {str(e)}")

async def main():
    # Ensure test files exist
    test_files = [
        ("shared/sample_inputs/urgent_complaint.eml", "email"),
        ("shared/sample_inputs/high_value_transaction.json", "json"),
        ("shared/sample_inputs/invoice_content.txt", "pdf")  # Note: You'll need to convert this to PDF
    ]
    
    async with httpx.AsyncClient() as client:
        for file_path, expected_format in test_files:
            if Path(file_path).exists():
                await test_upload_file(client, Path(file_path), expected_format)
            else:
                print(f"\n⚠️ Warning: Test file not found: {file_path}")

if __name__ == "__main__":
    print("🚀 Starting system test...")
    asyncio.run(main()) 