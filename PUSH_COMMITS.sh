#!/bin/bash
# Manual push script for quantum tarot commits
# Run this when ready to push to GitHub

echo "🔮 Pushing Quantum Tarot commits to GitHub..."
echo ""
echo "Branch: claude/quantum-tarot-app-setup-011CV4XWLj8y1V5TvBkRgz5M"
echo "Commits to push: 2"
echo ""
echo "Commit 1: Card database + AGI query engine + flip UI"
echo "Commit 2: Dual version system (Free vs Premium)"
echo ""

cd /home/user/HungryOrca

echo "Attempting push..."
git push -u origin claude/quantum-tarot-app-setup-011CV4XWLj8y1V5TvBkRgz5M

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Commits pushed to GitHub."
else
    echo ""
    echo "❌ Push failed. Try these alternatives:"
    echo ""
    echo "Option 1: Push via GitHub CLI"
    echo "  gh auth login"
    echo "  git push"
    echo ""
    echo "Option 2: Create PR from web UI"
    echo "  Visit: https://github.com/aphoticshaman/HungryOrca"
    echo "  The branch exists, just needs commits pushed"
    echo ""
    echo "Option 3: Check session/permissions"
    echo "  git remote -v"
    echo "  git config --list | grep user"
    echo ""
fi
