import zipfile
import os

def zip_project(output_filename):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add requirements.txt
        if os.path.exists('requirements.txt'):
            zipf.write('requirements.txt')
            
        # Add src folder
        for root, dirs, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path)
                    
        # Add data folder (only csv)
        for root, dirs, files in os.walk('data'):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path)

        # Add validation script
        if os.path.exists('validate_up_model.py'):
            zipf.write('validate_up_model.py')

    print(f"Created {output_filename}")

if __name__ == "__main__":
    zip_project('project_code.zip')
