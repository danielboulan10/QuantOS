# Security policy

## Scope

QuantOS is a research library and a static website. It holds no user accounts, no
credentials and no personal data, and the published site runs no server-side code
— it is pre-generated files on GitHub Pages.

The realistic concerns are therefore narrower than for most applications:

- **Untrusted input to the parsers.** `data/loader.py`, `data/options.py`,
  `data/intraday.py` and `data/market.py` read files and network responses. A
  crash, hang or unbounded memory use on a malformed input is a bug worth
  reporting.
- **The local viewer.** `quantos serve` binds to localhost and has no
  authentication, deliberately — it is a local research tool. Exposing it on a
  public interface would publish an unauthenticated endpoint that makes outbound
  requests on the operator's behalf, and the module docstring says so. Reports
  that it is insecure *when deliberately exposed* are expected behaviour, not
  vulnerabilities.
- **Dependency risk**, which is small by construction: the runtime depends on
  NumPy alone ([DDR-002](docs/ddr/DDR-002-numpy-only-runtime.md)).

## Reporting

Open a [security advisory](https://github.com/danielboulan10/QuantOS/security/advisories/new),
or a normal issue if the finding is not sensitive. Please include a reproduction.

I am one person and this is not a funded project, so I cannot promise a response
window. I will acknowledge what I can.

## Not a security issue

- **The numbers being wrong.** That is a correctness bug and belongs in a normal
  issue — a valuable one, and better with a numerical counterexample.
- **Trading on the output and losing money.** Nothing here is investment advice,
  the site says so on every page, and the analysis describes distributions rather
  than outcomes.
