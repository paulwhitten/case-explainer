# Deploying Documentation to GitHub Pages

## Automatic Deployment (Recommended)

The repository is configured to automatically build and deploy documentation to GitHub Pages when you push to the main branch.

### Setup Steps

1. **Enable GitHub Pages in your repository:**
   - Go to your GitHub repository
   - Settings → Pages
   - Source: Select "GitHub Actions"

2. **Push your changes:**
   ```bash
   git add docs/index.rst docs/api    # name the files you changed
   git commit -m "Update documentation"
   git push origin main
   ```

3. **Wait for the workflow to complete:**
   - Go to the "Actions" tab in your repository
   - Watch the "Build and Deploy Documentation" workflow run
   - Once complete, your docs will be available at:
     `https://USERNAME.github.io/case-explainer/`

4. **Verify deployment:**
   - The workflow (`.github/workflows/docs.yml`) uploads the built site as a
     Pages artifact and deploys it; no `gh-pages` branch is involved
   - Documentation is rebuilt and deployed on every push to main
   - The build runs with `-W`, so any Sphinx warning fails the deployment

## Manual Deployment (Alternative)

Run the workflow by hand from the Actions tab ("Build and Deploy
Documentation" → "Run workflow"). It builds the current `main` branch.

## Updating Documentation

Simply push changes to your main branch and the docs will automatically rebuild:

```bash
# Make changes to .rst files or docstrings, then add those files by name.
# Avoid `git add docs/`, which also stages any untracked notes in the folder.
git add docs/api/explainer.rst
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
- Verify Sphinx builds locally with `sphinx-build -W -b html docs docs/_build/html`

**404 on GitHub Pages:**
- Wait a few minutes after the first deployment
- Check Settings → Pages to verify the source is "GitHub Actions"
- Ensure the workflow completed successfully in the Actions tab

**Documentation not updating:**
- Clear your browser cache
- Verify the workflow ran after your push
- Open the workflow run and confirm the deploy job finished

## Local Preview

Before pushing, preview docs locally:

```bash
cd docs
make html
python3 -m http.server 8000 --directory _build/html
# Open http://localhost:8000
```
