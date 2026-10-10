# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/ewhauser/outis/security/advisories/new).
Do not open a public issue.

Include the affected version or commit, reproduction steps, and the impact.
**Never include real protected health information.** Use synthetic text that
reproduces the problem.

## What counts as a security issue

- PHI appearing in logs, error messages, response metadata or caches.
- Authentication or binding bypasses in the HTTP API.
- Recovery of replacement secrets, patient anchors or date offsets from
  output.
- Parsing vulnerabilities in structured or file input (XML, CSV, JSON or text
  files).

A missed detection, where PHI is left in the output, is a model accuracy
limitation and not a vulnerability. Please report it as a regular issue,
using synthetic text that shows the pattern.

## Deployment guidance

See [deployment](docs/deployment.md) for the built-in controls: loopback
binding, API keys, no request logging, and `Cache-Control: no-store`. TLS,
network policy and log handling are the deployer's responsibility.
