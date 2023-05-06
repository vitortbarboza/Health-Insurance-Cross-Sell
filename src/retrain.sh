echo "Executing retrain... "
python3 retrain/retrain.py

echo "Adding files..."
cd src
git add .
git commit -m "last retrain"

echo "Pushing to remote repository..."
git push origin main


