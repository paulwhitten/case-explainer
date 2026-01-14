# Deploying Documentation to GitHub Pages

## Automatic Deployment (Recommended)

The repository is configured to automatically build and deploy documentation to GitHub Pages when you push to the main branch.

### Setup Steps

1. **Enable GitHub Pages in your repository:**
   - Go to your GitHub repository
   - Settings → Pages
   - Source: Select "Deploy from a branch"
   - Branch: Select `gh-pages` (will be created automatically by the workflow)
   - Click Save

2. **Push your changes:**
   ```bash
   git add .
   git commit -m "Add Sphinx documentation and GitHub Actions workflow"
   git push origin main
   ```

3. **Wait for the workflow to complete:**
   - Go to the "Actions" tab in your repository
   - Watch the "Build and Deploy Documentation" workflow run
   - Once complete, your docs will be available at:
     `https://USERNAME.github.io/case-explainer/`

4. **Verify deployment:**
   - The workflow creates a `gh-pages` branch automatically
   - Documentation is rebuilt and deployed on every push to main

## Manual Deployment (Alternative)

If you prefer to deploy manually:

```bash
# Build the documentation
cd docs
make html

# Install ghp-import (one-time)
pip install ghp-import

# Deploy to gh-pages branch
ghp-import -n -p -f _build/html
```

## Updating Documentation

Simply push changes to your main branch and the docs will automatically rebuild:

```bash
# Make changes to .rst files or docstrings
git add docs/
git commit -m "Update documentation"
git push origin main

# GitHub Actions will automatically rebuild and deploy
```

## Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file to `docs/`:
   ```bash
   echo "docs.yourdomain.com" > docs/CNAME
   ```

2. Configure DNS with your domain provider:
   - Add a CNAME record pointing to `USERNAME.github.io`

3. Update repository settings:
   - Settings → Pages → Custom domain
   - Enter your domain and save

## Troubleshooting

**Workflow fails:**
- Check the Actions tab for error messages
- Ensure all dependencies are listed in the workflow
- Verify Sphinx builds locally with `cd docs && make html`

**404 on GitHub Pages:**
- Wait a few minutes after the first deployment
- Check Settings → Pages to verify the branch is set correctly
- Ensure the workflow completed successfully in the Actions tab

**Documentation not updating:**
- Clear your browser cache
- Verify the workflow ran after your push
- Check the `gh-pages` branch to see if files were updated

## Local Preview

Before pushing, preview docs locally:

```bash
cd docs
make html
python3 -m http.server 8000 --directory _build/html
# Open http://localhost:8000
```
