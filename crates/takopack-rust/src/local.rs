use crate::rpm::{self, RpmPackageInfo, SourceArchiveOverride};
use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Component, Path, PathBuf};
use takopack_core::config::Config;
use takopack_core::util::write_file_ensuring_dir;
use tar::Archive;
use toml::Value;

use crate::crates::CrateInfo;
use crate::package::PackageExecuteArgs;
use crate::range_audit::{self, RangeCapabilityPolicy};

/// Process a local crate directory and generate spec file
pub fn process_local_package(
    path: &Path,
    output_dir: Option<PathBuf>,
    finish_args: PackageExecuteArgs,
    range_capability_policy: RangeCapabilityPolicy,
) -> Result<()> {
    process_local_package_with_source(path, output_dir, finish_args, range_capability_policy, None)
}

/// Process a local crate using exact dependency versions from Cargo.lock.
pub fn process_local_package_with_lockfile(
    path: &Path,
    output_dir: Option<PathBuf>,
    mut finish_args: PackageExecuteArgs,
    range_capability_policy: RangeCapabilityPolicy,
    lockfile: Option<&Path>,
) -> Result<()> {
    finish_args.lockfile = lockfile.map(Path::to_path_buf);
    process_local_package_with_source(path, output_dir, finish_args, range_capability_policy, None)
}

/// Process a local crate while recovering a reproducible Git source from the
/// workspace Cargo.lock that selected it.
///
/// `source_archive` is an optional already-downloaded copy of the archive. It
/// is useful for offline/repeatable generation; when omitted, TakoPack fetches
/// the archive URL derived from the locked Git source.
pub fn process_local_package_with_source(
    path: &Path,
    output_dir: Option<PathBuf>,
    finish_args: PackageExecuteArgs,
    range_capability_policy: RangeCapabilityPolicy,
    source_archive: Option<&Path>,
) -> Result<()> {
    if source_archive.is_some() && finish_args.lockfile.is_none() {
        anyhow::bail!("--source-archive requires --lockfile");
    }

    // Canonicalize the path first to get absolute path
    let path_abs =
        fs::canonicalize(path).with_context(|| format!("Failed to resolve path: {:?}", path))?;

    // Determine the crate directory and Cargo.toml path
    let cargo_toml = if path_abs.is_file() {
        // Path is a .toml file
        if !path_abs.extension().map(|e| e == "toml").unwrap_or(false) {
            anyhow::bail!("File must be a .toml file: {:?}", path_abs);
        }
        path_abs
    } else if path_abs.is_dir() {
        // Path is a directory
        let toml = path_abs.join("Cargo.toml");
        if !toml.exists() {
            anyhow::bail!("Cargo.toml not found in directory: {:?}", path_abs);
        }
        toml
    } else {
        anyhow::bail!(
            "Invalid path: must be a directory or Cargo.toml file: {:?}",
            path_abs
        );
    };

    log::info!("Processing local crate from: {:?}", cargo_toml);

    let temp_crate_dir =
        tempfile::tempdir().context("Failed to create temporary crate directory")?;
    let temp_cargo_toml =
        materialize_manifest_backed_temp_crate(&cargo_toml, temp_crate_dir.path())?;

    log::info!(
        "Temporary crate structure created at: {:?}",
        temp_crate_dir.path()
    );

    // Now process this temporary complete crate with full takopack pipeline
    process_complete_crate(
        temp_crate_dir.path(),
        &temp_cargo_toml,
        output_dir,
        finish_args,
        range_capability_policy,
        source_archive,
    )
}

pub(crate) fn materialize_manifest_backed_temp_crate(
    cargo_toml: &Path,
    temp_dir: &Path,
) -> Result<PathBuf> {
    let cargo_toml_content = fs::read_to_string(cargo_toml)
        .with_context(|| format!("Failed to read Cargo.toml: {:?}", cargo_toml))?;
    let manifest: Value = toml::from_str(&cargo_toml_content)
        .with_context(|| format!("Failed to parse Cargo.toml: {:?}", cargo_toml))?;

    let temp_cargo_toml = temp_dir.join("Cargo.toml");
    fs::write(&temp_cargo_toml, cargo_toml_content).with_context(|| {
        format!(
            "Failed to write temporary Cargo.toml: {:?}",
            temp_cargo_toml
        )
    })?;

    if let Some(parent) = cargo_toml.parent() {
        let config = parent.join("takopack.toml");
        if config.exists() {
            fs::copy(&config, temp_dir.join("takopack.toml"))
                .with_context(|| format!("Failed to copy takopack.toml from {:?}", config))?;
        }
    }

    materialize_manifest_paths(&manifest, temp_dir)?;
    Ok(temp_cargo_toml)
}

fn materialize_manifest_paths(manifest: &Value, root: &Path) -> Result<()> {
    let mut files = BTreeSet::new();
    let mut explicit_targets = 0usize;
    let mut has_lib_target = false;
    let mut autolib = true;

    if let Some(package) = manifest.get("package").and_then(Value::as_table) {
        if let Some(value) = package.get("autolib").and_then(Value::as_bool) {
            autolib = value;
        }

        if let Some(build) = package.get("build") {
            match build {
                Value::Boolean(false) => {}
                Value::String(path) => {
                    files.insert(path.clone());
                }
                _ => {
                    files.insert("build.rs".to_string());
                }
            }
        }

        for key in ["readme", "license-file"] {
            if let Some(path) = package.get(key).and_then(Value::as_str) {
                files.insert(path.to_string());
            }
        }

        if let Some(include) = package.get("include").and_then(Value::as_array) {
            for item in include.iter().filter_map(Value::as_str) {
                let item = normalize_package_root_relative_path(item);
                if should_materialize_include(&item) {
                    files.insert(item.trim_end_matches('/').to_string());
                }
            }
        }
    }

    if let Some(lib) = manifest.get("lib").and_then(Value::as_table) {
        explicit_targets += 1;
        has_lib_target = true;
        files.insert(
            lib.get("path")
                .and_then(Value::as_str)
                .unwrap_or("src/lib.rs")
                .to_string(),
        );
    }

    for (key, default_dir) in [
        ("bin", "src/bin"),
        ("example", "examples"),
        ("test", "tests"),
        ("bench", "benches"),
    ] {
        if let Some(targets) = manifest.get(key).and_then(Value::as_array) {
            for target in targets.iter().filter_map(Value::as_table) {
                explicit_targets += 1;
                let path = target
                    .get("path")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| {
                        target
                            .get("name")
                            .and_then(Value::as_str)
                            .map(|name| format!("{}/{}.rs", default_dir, name))
                    })
                    .unwrap_or_else(|| default_target_path(key).to_string());
                files.insert(path);
            }
        }
    }

    if !has_lib_target && autolib {
        files.insert("src/lib.rs".to_string());
    }

    if explicit_targets == 0 {
        files.insert("src/lib.rs".to_string());
    }

    let files: Vec<_> = files
        .iter()
        .filter(|path| !has_materialized_child_path(path, &files))
        .cloned()
        .collect();

    for path in files {
        write_placeholder_file(root, &path)?;
    }

    Ok(())
}

fn default_target_path(kind: &str) -> &'static str {
    match kind {
        "bin" => "src/main.rs",
        "example" => "examples/example.rs",
        "test" => "tests/test.rs",
        "bench" => "benches/bench.rs",
        _ => "src/lib.rs",
    }
}

fn should_materialize_include(path: &str) -> bool {
    !path.starts_with('!')
        && !path.contains('*')
        && !path.contains('?')
        && !path.contains('[')
        && path != "Cargo.toml"
}

fn normalize_package_root_relative_path(path: &str) -> String {
    path.trim_start_matches('/').to_string()
}

fn has_materialized_child_path(path: &str, files: &BTreeSet<String>) -> bool {
    let path = Path::new(path);
    files
        .iter()
        .any(|other| Path::new(other) != path && Path::new(other).starts_with(path))
}

fn write_placeholder_file(root: &Path, relative_path: &str) -> Result<()> {
    let relative = safe_manifest_relative_path(relative_path)?;
    let path = root.join(relative);
    if path.exists() {
        return Ok(());
    }

    let content = match path.extension().and_then(|ext| ext.to_str()) {
        Some("rs") => "// Placeholder for takopack localpkg spec generation.\n",
        Some("md") => "# Placeholder\n",
        _ => "Placeholder for takopack localpkg spec generation.\n",
    };
    write_file_ensuring_dir(&path, content)
}

fn safe_manifest_relative_path(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    if path.is_absolute() {
        anyhow::bail!("Cargo.toml path must be relative: {:?}", path);
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        anyhow::bail!("Cargo.toml path escapes the temporary crate: {:?}", path);
    }
    Ok(path.to_path_buf())
}

/// Process a complete crate directory (with src/) using full takopack pipeline
fn process_complete_crate(
    temp_crate_dir: &Path,
    cargo_toml: &Path,
    output_dir: Option<PathBuf>,
    mut finish_args: PackageExecuteArgs,
    range_capability_policy: RangeCapabilityPolicy,
    source_archive: Option<&Path>,
) -> Result<()> {
    // Load config if available
    let config_path = temp_crate_dir.join("takopack.toml");
    let (config_path, config) = if config_path.exists() {
        let config = Config::parse(&config_path).context("failed to parse takopack.toml")?;
        (Some(config_path), config)
    } else {
        Config::load()?
    };

    // Create CrateInfo from local crate (now it has src/ so Cargo APIs will work)
    let mut crate_info = CrateInfo::new_with_local_crate_from_path(cargo_toml)
        .with_context(|| format!("Failed to load crate from: {:?}", cargo_toml))?;

    let crate_name = crate_info.crate_name();
    // It's a full version,like "0.9.11+spec-1.1.0"
    let version = crate_info.version();

    log::info!("Crate: {} {}", crate_name, version);

    if let Some(lockfile) = finish_args.lockfile.as_deref() {
        finish_args.lockfile_deps =
            lockfile_dependencies_for_package(lockfile, crate_name, &version.to_string())?;
    }

    // Create RpmPackageInfo
    let rpm_info =
        RpmPackageInfo::new(&crate_info, env!("CARGO_PKG_VERSION"), config.semver_suffix);

    let output_names = takopack_core::util::rust_crate_output_names(crate_name, version);

    let source_archive_override = match finish_args.lockfile.as_deref() {
        Some(lock) => git_archive_source_from_lockfile(lock, crate_name, version, source_archive)?,
        None => None,
    };

    if range_capability_policy != RangeCapabilityPolicy::Allow {
        let mut warnings = range_audit::audit_cargo_dependencies(
            crate_info.dependencies(),
            Some(&output_names.directory),
        );
        if let Some(lockfile_deps) = finish_args.lockfile_deps.as_ref() {
            warnings.retain(|warning| {
                let dash_name = warning.dependency.replace('_', "-");
                !lockfile_deps.contains_key(&warning.dependency)
                    && !lockfile_deps.contains_key(&dash_name)
            });
        }
        if range_audit::emit_warnings(&warnings, range_capability_policy) {
            anyhow::bail!("range capability audit failed (policy: error)");
        }
    }

    // Determine final output package directory.
    let final_output =
        takopack_core::util::package_final_output_dir(output_dir.as_deref(), &output_names)?;

    fs::create_dir_all(&final_output)
        .with_context(|| format!("Failed to create output directory: {:?}", final_output))?;

    // Create a temporary directory for takopack processing
    let tempdir =
        tempfile::tempdir_in(temp_crate_dir).context("Failed to create temporary directory")?;

    log::info!("Tempdir created at: {:?}", tempdir.path());
    log::info!("Preparing takopack folder");

    // Apply overrides and generate spec file
    let prepare_result = rpm::prepare_takopack_folder(
        &mut crate_info,
        &rpm_info,
        config_path.as_deref(),
        &config,
        temp_crate_dir,
        &tempdir,
        finish_args.changelog_ready,
        finish_args.copyright_guess_harder,
        !finish_args.no_overlay_write_back,
        None, // TODO: sha256: local packages don't have downloaded crate files, maybe consider record the sha256 when use pkg.
        source_archive_override,
        finish_args.lockfile_deps, // Pass lockfile dependencies if available
        finish_args.with_spdx,
    );

    if let Err(e) = &prepare_result {
        log::error!("prepare_takopack_folder failed: {:?}", e);
    }
    prepare_result?;

    // Note: prepare_takopack_folder renames tempdir to output_dir/takopack
    let takopack_dir = temp_crate_dir.join("takopack");
    log::info!("Takopack folder should be at: {:?}", takopack_dir);
    log::info!("Takopack dir exists: {}", takopack_dir.exists());

    // Copy spec file to output directory
    let source_spec = takopack_dir.join(&output_names.spec_file);
    let final_spec = final_output.join(&output_names.spec_file);

    // List files in takopack dir for debugging
    log::debug!("Listing files in takopack dir: {:?}", takopack_dir);
    match fs::read_dir(&takopack_dir) {
        Ok(entries) => {
            for entry in entries.flatten() {
                log::debug!("  - {:?}", entry.file_name());
            }
        }
        Err(e) => {
            log::error!("Failed to read takopack dir: {:?}", e);
        }
    }

    if source_spec.exists() {
        fs::copy(&source_spec, &final_spec)
            .with_context(|| format!("Failed to copy spec file to: {:?}", final_spec))?;
        takopack_core::util::copy_normalized_cargo_toml_to_dir(temp_crate_dir, &final_output)?;

        log::info!("Spec file saved to: {}", final_spec.display());
        println!("Spec file: {}", final_spec.display());
    } else {
        anyhow::bail!("Spec file not found at: {:?}", source_spec);
    }

    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum LockedGitReference {
    Rev,
    Tag(String),
    Commit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LockedGitSource {
    repository: String,
    repository_name: String,
    commit: String,
    reference: LockedGitReference,
}

fn git_archive_source_from_lockfile(
    lockfile: &Path,
    crate_name: &str,
    version: &semver::Version,
    source_archive: Option<&Path>,
) -> Result<Option<SourceArchiveOverride>> {
    let content = fs::read_to_string(lockfile)
        .with_context(|| format!("failed to read Cargo.lock: {}", lockfile.display()))?;
    let lock: Value = toml::from_str(&content)
        .with_context(|| format!("failed to parse Cargo.lock: {}", lockfile.display()))?;
    let packages = lock
        .get("package")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Cargo.lock has no [[package]] entries"))?;

    let mut sources = packages.iter().filter_map(|package| {
        let package = package.as_table()?;
        if package.get("name")?.as_str()? != crate_name
            || package.get("version")?.as_str()? != version.to_string()
        {
            return None;
        }
        package
            .get("source")?
            .as_str()
            .filter(|source| source.starts_with("git+"))
    });
    let Some(source) = sources.next() else {
        if source_archive.is_some() {
            anyhow::bail!(
                "--source-archive requires a Git source for {} {} in Cargo.lock",
                crate_name,
                version
            );
        }
        return Ok(None);
    };
    if sources.next().is_some() {
        anyhow::bail!(
            "Cargo.lock has multiple Git sources for {} {}; source selection is ambiguous",
            crate_name,
            version
        );
    }

    let source = parse_locked_github_source(source)?;
    let (source_url, fetch_url, source_macros) = match &source.reference {
        LockedGitReference::Tag(tag) => (
            format!(
                "{}/archive/refs/tags/%{{git_tag}}.tar.gz#/%{{crate_name}}-%{{git_tag}}.tar.gz",
                source.repository
            ),
            format!("{}/archive/refs/tags/{}.tar.gz", source.repository, tag),
            vec![
                ("git_tag".to_string(), tag.clone()),
                ("git_commit".to_string(), source.commit.clone()),
            ],
        ),
        LockedGitReference::Rev => (
            format!(
                "{}/archive/%{{git_commit}}.tar.gz#/%{{crate_name}}-%{{git_commit}}.tar.gz",
                source.repository
            ),
            format!("{}/archive/{}.tar.gz", source.repository, source.commit),
            vec![("git_commit".to_string(), source.commit.clone())],
        ),
        LockedGitReference::Commit => (
            format!(
                "{}/archive/%{{git_commit}}.tar.gz#/%{{crate_name}}-%{{git_commit}}.tar.gz",
                source.repository
            ),
            format!("{}/archive/{}.tar.gz", source.repository, source.commit),
            vec![("git_commit".to_string(), source.commit.clone())],
        ),
    };

    let bytes = match source_archive {
        Some(path) => fs::read(path)
            .with_context(|| format!("failed to read source archive: {}", path.display()))?,
        None => download_archive(&fetch_url)?,
    };
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let prep_dir = archive_top_level_directory(&bytes)?;
    if !prep_dir.starts_with(&format!("{}-", source.repository_name)) {
        anyhow::bail!(
            "Git archive top-level directory {:?} does not match repository {:?}",
            prep_dir,
            source.repository_name
        );
    }

    Ok(Some(SourceArchiveOverride {
        source_macros,
        source_url,
        sha256,
        prep_dir,
    }))
}

fn parse_locked_github_source(source: &str) -> Result<LockedGitSource> {
    let source = source
        .strip_prefix("git+")
        .ok_or_else(|| anyhow::anyhow!("not a Cargo Git source: {source}"))?;
    let (repository_and_query, commit) = source
        .rsplit_once('#')
        .ok_or_else(|| anyhow::anyhow!("Cargo Git source has no resolved commit: {source}"))?;
    if commit.len() != 40 || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("Cargo Git source has an invalid resolved commit: {commit}");
    }
    let (repository, query) = repository_and_query
        .split_once('?')
        .map_or((repository_and_query, ""), |(repository, query)| {
            (repository, query)
        });
    let repository = repository.trim_end_matches('/').trim_end_matches(".git");
    let github_path = repository.strip_prefix("https://github.com/").ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported Git host in {repository}; localpkg Git archives currently require an HTTPS GitHub source"
        )
    })?;
    let mut path_parts = github_path.split('/');
    let owner = path_parts.next().filter(|part| !part.is_empty());
    let repository_name = path_parts.next().filter(|part| !part.is_empty());
    if owner.is_none() || repository_name.is_none() || path_parts.next().is_some() {
        anyhow::bail!("invalid GitHub repository URL: {repository}");
    }
    let repository_name = repository_name.unwrap().to_string();

    let mut rev = None;
    let mut tag = None;
    let mut branch = None;
    for item in query.split('&').filter(|item| !item.is_empty()) {
        let Some((key, value)) = item.split_once('=') else {
            continue;
        };
        validate_git_selector(value)?;
        match key {
            "rev" => rev = Some(value.to_string()),
            "tag" => tag = Some(value.to_string()),
            "branch" => branch = Some(value.to_string()),
            _ => {}
        }
    }
    if rev.is_some() && tag.is_some() {
        anyhow::bail!("Cargo Git source contains both rev and tag selectors: {source}");
    }
    let reference = if rev.is_some() {
        LockedGitReference::Rev
    } else if let Some(tag) = tag {
        LockedGitReference::Tag(tag)
    } else {
        if branch.is_some() {
            log::warn!(
                "Cargo.lock selected a branch; using its resolved commit for a reproducible archive"
            );
        }
        LockedGitReference::Commit
    };

    Ok(LockedGitSource {
        repository: repository.to_string(),
        repository_name,
        commit: commit.to_ascii_lowercase(),
        reference,
    })
}

fn validate_git_selector(selector: &str) -> Result<()> {
    if selector.is_empty()
        || selector
            .chars()
            .any(|character| character.is_whitespace() || character == '%' || character == '#')
    {
        anyhow::bail!("unsupported Git selector in Cargo.lock: {selector:?}");
    }
    Ok(())
}

fn download_archive(url: &str) -> Result<Vec<u8>> {
    log::info!("downloading Git source archive: {url}");
    let response = ureq::AgentBuilder::new()
        .redirects(10)
        .build()
        .get(url)
        .call()
        .with_context(|| format!("failed to download Git source archive: {url}"))?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read Git source archive: {url}"))?;
    Ok(bytes)
}

fn archive_top_level_directory(bytes: &[u8]) -> Result<String> {
    let mut archive = Archive::new(GzDecoder::new(Cursor::new(bytes)));
    let mut top_level = None;
    let mut entries = 0usize;
    for entry in archive
        .entries()
        .context("failed to read Git source archive")?
    {
        let entry = entry.context("failed to read Git source archive entry")?;
        let entry_type = entry.header().entry_type();
        if entry_type.is_pax_global_extensions()
            || entry_type.is_pax_local_extensions()
            || entry_type.is_gnu_longname()
            || entry_type.is_gnu_longlink()
        {
            continue;
        }
        let path = entry
            .path()
            .context("failed to read Git source archive path")?;
        let first = path
            .components()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Git source archive contains an empty path"))?;
        let Component::Normal(first) = first else {
            anyhow::bail!(
                "Git source archive contains an unsafe path: {}",
                path.display()
            );
        };
        let first = first
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Git source archive path is not UTF-8"))?;
        match &top_level {
            Some(existing) if existing != first => anyhow::bail!(
                "Git source archive has multiple top-level entries: {:?} and {:?}",
                existing,
                first
            ),
            None => top_level = Some(first.to_string()),
            _ => {}
        }
        entries += 1;
    }
    if entries == 0 {
        anyhow::bail!("Git source archive is empty");
    }
    top_level.ok_or_else(|| anyhow::anyhow!("Git source archive has no top-level directory"))
}

pub fn lockfile_dependencies_for_package(
    lockfile: &Path,
    package_name: &str,
    package_version: &str,
) -> Result<Option<HashMap<String, semver::Version>>> {
    let content = fs::read_to_string(lockfile)
        .with_context(|| format!("Failed to read Cargo.lock: {:?}", lockfile))?;
    let document: Value = toml::from_str(&content)
        .with_context(|| format!("Failed to parse Cargo.lock: {:?}", lockfile))?;
    let packages = document
        .get("package")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Cargo.lock has no package array: {:?}", lockfile))?;

    let current = packages.iter().find(|package| {
        package.get("name").and_then(Value::as_str) == Some(package_name)
            && package.get("version").and_then(Value::as_str) == Some(package_version)
    });
    let Some(current) = current else {
        log::warn!(
            "Cargo.lock {:?} has no package {} {}; keeping Cargo.toml dependency ranges",
            lockfile,
            package_name,
            package_version
        );
        return Ok(None);
    };

    let Some(dependencies) = current.get("dependencies").and_then(Value::as_array) else {
        return Ok(Some(HashMap::new()));
    };

    let mut selected = HashMap::new();
    let mut conflicts = BTreeSet::new();
    for dependency in dependencies.iter().filter_map(Value::as_str) {
        let mut fields = dependency.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let explicit_version = fields
            .next()
            .and_then(|field| semver::Version::parse(field).ok());
        let candidates: Vec<_> = packages
            .iter()
            .filter(|package| package.get("name").and_then(Value::as_str) == Some(name))
            .filter_map(|package| {
                let version = package.get("version").and_then(Value::as_str)?;
                semver::Version::parse(version).ok()
            })
            .filter(|version| {
                explicit_version
                    .as_ref()
                    .is_none_or(|expected| version == expected)
            })
            .collect();

        if candidates.len() != 1 {
            log::warn!(
                "Cargo.lock dependency {:?} of {} {} resolves to {} package entries; keeping its Cargo.toml range",
                dependency,
                package_name,
                package_version,
                candidates.len()
            );
            continue;
        }

        let version = candidates[0].clone();
        match selected.get(name) {
            Some(existing) if existing != &version => {
                conflicts.insert(name.to_string());
            }
            None => {
                selected.insert(name.to_string(), version);
            }
            _ => {}
        }
    }
    for name in conflicts {
        selected.remove(&name);
        log::warn!(
            "Cargo.lock selects multiple versions of dependency {} for {} {}; keeping its Cargo.toml range",
            name,
            package_name,
            package_version
        );
    }
    Ok(Some(selected))
}

#[cfg(test)]
mod tests {
    use super::{
        archive_top_level_directory, lockfile_dependencies_for_package,
        materialize_manifest_backed_temp_crate, process_local_package,
        process_local_package_with_lockfile, process_local_package_with_source,
    };
    use crate::package::PackageExecuteArgs;
    use crate::range_audit::RangeCapabilityPolicy;

    use flate2::{Compression, write::GzEncoder};
    use semver::Version;
    use std::fs;
    use takopack_core::util::rust_crate_output_names;
    use tar::{Builder, EntryType, Header};

    fn write_test_git_archive(path: &std::path::Path, root: &str) {
        let file = fs::File::create(path).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut archive = Builder::new(encoder);
        let pax_content = b"19 comment=fixture\n";
        let mut pax_header = Header::new_gnu();
        pax_header.set_path("pax_global_header").unwrap();
        pax_header.set_entry_type(EntryType::XGlobalHeader);
        pax_header.set_size(pax_content.len() as u64);
        pax_header.set_mode(0o644);
        pax_header.set_mtime(0);
        pax_header.set_cksum();
        archive.append(&pax_header, &pax_content[..]).unwrap();
        let content = b"[package]\nname = \"fixture\"\nversion = \"0.0.0\"\n";
        let mut header = Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o644);
        header.set_mtime(0);
        header.set_cksum();
        archive
            .append_data(&mut header, format!("{root}/Cargo.toml"), &content[..])
            .unwrap();
        archive.finish().unwrap();
        archive.into_inner().unwrap().finish().unwrap();
    }

    #[test]
    fn git_archive_top_level_ignores_pax_global_header() {
        let temp = tempfile::tempdir().unwrap();
        let archive = temp.path().join("source.tar.gz");
        write_test_git_archive(&archive, "project-0123456789abcdef");

        assert_eq!(
            archive_top_level_directory(&fs::read(archive).unwrap()).unwrap(),
            "project-0123456789abcdef"
        );
    }

    #[test]
    fn localpkg_materializes_declared_manifest_paths() {
        let source = tempfile::tempdir().unwrap();
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            source.path().join("Cargo.toml"),
            r#"
[package]
name = "shape"
version = "1.2.3"
edition = "2021"
build = "build/main.rs"
readme = "docs/README.md"
license-file = "licenses/LICENSE.txt"
include = ["NOTICE"]

[lib]
path = "src/shape/lib.rs"

[[bin]]
name = "shape-cli"
path = "cli/main.rs"

[[example]]
name = "demo"
"#,
        )
        .unwrap();
        fs::write(source.path().join("takopack.toml"), "[source]\n").unwrap();

        materialize_manifest_backed_temp_crate(&source.path().join("Cargo.toml"), temp.path())
            .unwrap();

        for path in [
            "Cargo.toml",
            "takopack.toml",
            "build/main.rs",
            "docs/README.md",
            "licenses/LICENSE.txt",
            "NOTICE",
            "src/shape/lib.rs",
            "cli/main.rs",
            "examples/demo.rs",
        ] {
            assert!(temp.path().join(path).exists(), "missing {path}");
        }
    }

    #[test]
    fn localpkg_adds_default_lib_for_manifest_without_targets() {
        let source = tempfile::tempdir().unwrap();
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            source.path().join("Cargo.toml"),
            r#"
[package]
name = "minimal"
version = "0.1.0"
edition = "2021"
"#,
        )
        .unwrap();

        materialize_manifest_backed_temp_crate(&source.path().join("Cargo.toml"), temp.path())
            .unwrap();

        assert!(temp.path().join("src/lib.rs").exists());
    }

    #[test]
    fn localpkg_materializes_root_relative_includes() {
        let source = tempfile::tempdir().unwrap();
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            source.path().join("Cargo.toml"),
            r#"
[package]
name = "rooted"
version = "0.1.0"
edition = "2021"
include = [
    "/Cargo.toml",
    "/CHANGELOG.md",
    "/src",
]

[lib]
path = "src/lib.rs"
"#,
        )
        .unwrap();

        materialize_manifest_backed_temp_crate(&source.path().join("Cargo.toml"), temp.path())
            .unwrap();

        assert!(temp.path().join("CHANGELOG.md").exists());
        assert!(temp.path().join("src").is_dir());
        assert!(temp.path().join("src/lib.rs").exists());
    }

    #[test]
    fn localpkg_does_not_materialize_parent_include_as_file() {
        let source = tempfile::tempdir().unwrap();
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            source.path().join("Cargo.toml"),
            r#"
[package]
name = "benchy"
version = "0.1.0"
edition = "2021"
include = ["benches"]

[lib]
path = "src/lib.rs"

[[bench]]
name = "mod"
path = "benches/mod.rs"
"#,
        )
        .unwrap();

        materialize_manifest_backed_temp_crate(&source.path().join("Cargo.toml"), temp.path())
            .unwrap();

        assert!(temp.path().join("benches").is_dir());
        assert!(temp.path().join("benches/mod.rs").exists());
    }

    #[test]
    fn localpkg_materializes_auto_lib_with_other_targets() {
        let source = tempfile::tempdir().unwrap();
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            source.path().join("Cargo.toml"),
            r#"
[package]
name = "auto_lib"
version = "0.1.0"
edition = "2021"

[[test]]
name = "tests"

[[bench]]
name = "benchmarks"
"#,
        )
        .unwrap();

        materialize_manifest_backed_temp_crate(&source.path().join("Cargo.toml"), temp.path())
            .unwrap();

        assert!(temp.path().join("src/lib.rs").exists());
        assert!(temp.path().join("tests/tests.rs").exists());
        assert!(temp.path().join("benches/benchmarks.rs").exists());
    }

    #[test]
    fn localpkg_generates_spec_from_manifest_only_crate() {
        let source = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        fs::write(
            source.path().join("Cargo.toml"),
            r#"
[package]
name = "localpkg_smoke"
version = "0.1.0"
edition = "2021"
"#,
        )
        .unwrap();

        let finish = PackageExecuteArgs {
            changelog_ready: false,
            copyright_guess_harder: false,
            no_overlay_write_back: false,
            with_spdx: false,
            lockfile: None,
            lockfile_deps: None,
        };

        let output_names =
            rust_crate_output_names("localpkg_smoke", &Version::parse("0.1.0").unwrap());
        let output_root = output.path().join("explicit-output-root");
        let package_dir = output_root.join(&output_names.directory);

        process_local_package(
            source.path(),
            Some(output_root.clone()),
            finish,
            RangeCapabilityPolicy::Allow,
        )
        .unwrap();

        assert!(package_dir.join(&output_names.spec_file).exists());
        assert!(package_dir.join("Cargo.toml").exists());
        assert!(!output_root.join(&output_names.spec_file).exists());
    }

    #[test]
    fn localpkg_merges_features_with_the_same_normalized_rpm_name() {
        let source = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        fs::write(
            source.path().join("Cargo.toml"),
            r#"
[package]
name = "feature_collision"
version = "0.1.0"
edition = "2021"

[features]
__rustls = ["rustls"]
rustls-tls-manual-roots = ["__rustls"]

[dependencies]
rustls = { version = "0.21", optional = true }
"#,
        )
        .unwrap();

        let finish = PackageExecuteArgs {
            changelog_ready: false,
            copyright_guess_harder: false,
            no_overlay_write_back: false,
            with_spdx: false,
            lockfile: None,
            lockfile_deps: None,
        };

        let output_names =
            rust_crate_output_names("feature_collision", &Version::parse("0.1.0").unwrap());
        let output_root = output.path().join("explicit-output-root");

        process_local_package(
            source.path(),
            Some(output_root.clone()),
            finish,
            RangeCapabilityPolicy::Allow,
        )
        .unwrap();

        let spec = fs::read_to_string(
            output_root
                .join(&output_names.directory)
                .join(&output_names.spec_file),
        )
        .unwrap();
        assert_eq!(
            1,
            spec.lines()
                .filter(|line| *line == "%package     -n %{name}+rustls")
                .count(),
            "normalized feature names must identify a single RPM subpackage:\n{spec}"
        );
        assert!(spec.contains("Requires:       crate(rustls-0.21/default) >= 0.21.0"));
        assert!(
            spec.contains("Provides:       crate(%{pkgname}/rustls-tls-manual-roots) = %{version}")
        );
    }

    #[test]
    fn lockfile_selects_direct_dependency_version_for_current_package() {
        let temp = tempfile::tempdir().unwrap();
        let lockfile = temp.path().join("Cargo.lock");
        fs::write(
            &lockfile,
            r#"
version = 4

[[package]]
name = "consumer"
version = "1.2.3"
dependencies = [
 "itertools 0.14.0",
]

[[package]]
name = "itertools"
version = "0.10.5"

[[package]]
name = "itertools"
version = "0.14.0"
"#,
        )
        .unwrap();

        let selected = lockfile_dependencies_for_package(&lockfile, "consumer", "1.2.3")
            .unwrap()
            .unwrap();
        assert_eq!(
            selected.get("itertools"),
            Some(&Version::parse("0.14.0").unwrap())
        );
    }

    #[test]
    fn localpkg_uses_lock_selected_compat_capability() {
        let source = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        fs::write(
            source.path().join("Cargo.toml"),
            r#"
[package]
name = "range_consumer"
version = "1.2.3"
edition = "2021"

[dependencies]
itertools = ">=0.10.1, <=0.14"
"#,
        )
        .unwrap();
        fs::write(
            source.path().join("Cargo.lock"),
            r#"
version = 4

[[package]]
name = "range_consumer"
version = "1.2.3"
dependencies = [
 "itertools 0.14.0",
]

[[package]]
name = "itertools"
version = "0.10.5"

[[package]]
name = "itertools"
version = "0.14.0"
"#,
        )
        .unwrap();

        let finish = PackageExecuteArgs {
            changelog_ready: false,
            copyright_guess_harder: false,
            no_overlay_write_back: false,
            with_spdx: false,
            lockfile: None,
            lockfile_deps: None,
        };
        let output_names =
            rust_crate_output_names("range_consumer", &Version::parse("1.2.3").unwrap());
        let output_root = output.path().join("explicit-output-root");

        process_local_package_with_lockfile(
            source.path(),
            Some(output_root.clone()),
            finish,
            RangeCapabilityPolicy::Error,
            Some(&source.path().join("Cargo.lock")),
        )
        .unwrap();

        let spec = fs::read_to_string(
            output_root
                .join(&output_names.directory)
                .join(&output_names.spec_file),
        )
        .unwrap();
        assert!(spec.contains("Requires:       crate(itertools-0.14/default) >= 0.14.0"));
        assert!(!spec.contains("crate(itertools-0.10/"));
    }

    #[test]
    fn lockfile_does_not_rewrite_hyphen_prefixed_dependency() {
        let source = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        fs::write(
            source.path().join("Cargo.toml"),
            r#"
[package]
name = "prefix_consumer"
version = "1.2.3"
edition = "2021"

[dependencies]
tower = { version = "0.5.2", default-features = false, features = ["util"] }
tower-http = { version = "0.6.0", optional = true, default-features = false, features = ["cors"] }

[features]
docs = ["dep:tower-http"]
"#,
        )
        .unwrap();
        fs::write(
            source.path().join("Cargo.lock"),
            r#"
version = 4

[[package]]
name = "prefix_consumer"
version = "1.2.3"
dependencies = [
 "tower",
]

[[package]]
name = "tower"
version = "0.5.3"
"#,
        )
        .unwrap();

        let finish = PackageExecuteArgs {
            changelog_ready: false,
            copyright_guess_harder: false,
            no_overlay_write_back: false,
            with_spdx: false,
            lockfile: None,
            lockfile_deps: None,
        };
        let output_names =
            rust_crate_output_names("prefix_consumer", &Version::parse("1.2.3").unwrap());
        let output_root = output.path().join("explicit-output-root");

        process_local_package_with_lockfile(
            source.path(),
            Some(output_root.clone()),
            finish,
            RangeCapabilityPolicy::Error,
            Some(&source.path().join("Cargo.lock")),
        )
        .unwrap();

        let spec = fs::read_to_string(
            output_root
                .join(&output_names.directory)
                .join(&output_names.spec_file),
        )
        .unwrap();
        assert!(spec.contains("Requires:       crate(tower-0.5/util) >= 0.5.3"));
        assert!(spec.contains("Requires:       crate(tower-http-0.6/cors) >= 0.6.0"));
        assert!(!spec.contains("crate(tower-0.5/cors)"));
    }

    #[test]
    fn localpkg_generates_locked_git_sources_for_rev_and_tag() {
        struct Case<'a> {
            name: &'a str,
            version: &'a str,
            source: &'a str,
            archive_root: &'a str,
            expected_macros: &'a [&'a str],
            expected_source: &'a str,
        }

        let cases = [
            Case {
                name: "llm-multimodal",
                version: "1.7.1",
                source: "git+https://github.com/smg-project/llm-multimodal?rev=15adba5e025d8636ba4a334fb379b1371f6196a1#15adba5e025d8636ba4a334fb379b1371f6196a1",
                archive_root: "llm-multimodal-15adba5e025d8636ba4a334fb379b1371f6196a1",
                expected_macros: &["%global git_commit 15adba5e025d8636ba4a334fb379b1371f6196a1"],
                expected_source: "Source:         https://github.com/smg-project/llm-multimodal/archive/%{git_commit}.tar.gz#/%{crate_name}-%{git_commit}.tar.gz",
            },
            Case {
                name: "oss-harmony",
                version: "0.0.11",
                source: "git+https://github.com/oss-harmony/harmony?tag=v0.0.11#76e849426cc092f84509e31a17027755f67d662a",
                archive_root: "harmony-0.0.11",
                expected_macros: &[
                    "%global git_tag v0.0.11",
                    "%global git_commit 76e849426cc092f84509e31a17027755f67d662a",
                ],
                expected_source: "Source:         https://github.com/oss-harmony/harmony/archive/refs/tags/%{git_tag}.tar.gz#/%{crate_name}-%{git_tag}.tar.gz",
            },
        ];

        for case in cases {
            let source = tempfile::tempdir().unwrap();
            let output = tempfile::tempdir().unwrap();
            fs::write(
                source.path().join("Cargo.toml"),
                format!(
                    "[package]\nname = {:?}\nversion = {:?}\nedition = \"2021\"\n",
                    case.name, case.version
                ),
            )
            .unwrap();
            let lockfile = source.path().join("Cargo.lock");
            fs::write(
                &lockfile,
                format!(
                    "version = 4\n\n[[package]]\nname = {:?}\nversion = {:?}\nsource = {:?}\n",
                    case.name, case.version, case.source
                ),
            )
            .unwrap();
            let archive = source.path().join("source.tar.gz");
            write_test_git_archive(&archive, case.archive_root);

            let finish = PackageExecuteArgs {
                changelog_ready: false,
                copyright_guess_harder: false,
                no_overlay_write_back: false,
                with_spdx: true,
                lockfile: Some(lockfile.clone()),
                lockfile_deps: None,
            };
            process_local_package_with_source(
                source.path(),
                Some(output.path().to_path_buf()),
                finish,
                RangeCapabilityPolicy::Allow,
                Some(&archive),
            )
            .unwrap();

            let output_names =
                rust_crate_output_names(case.name, &Version::parse(case.version).unwrap());
            let spec = fs::read_to_string(
                output
                    .path()
                    .join(output_names.directory)
                    .join(output_names.spec_file),
            )
            .unwrap();
            for expected in case.expected_macros {
                assert!(spec.contains(expected), "missing {expected:?}:\n{spec}");
            }
            assert!(spec.contains(case.expected_source), "{spec}");
            assert!(
                spec.contains(&format!("BuildOption(prep):  -n {}", case.archive_root)),
                "{spec}"
            );
            assert!(!spec.contains("sha256:\n"), "{spec}");
            assert!(!spec.contains("static.crates.io"), "{spec}");
        }
    }
}
