# Cloudflare Setup for z86.dev

## 1. DNS Records

In Cloudflare dashboard for `z86.dev`:

| Type | Name | Content       | Proxy | TTL  |
|------|------|---------------|-------|------|
| A    | @    | <SERVER_IP>   | Yes   | Auto |
| A    | www  | <SERVER_IP>   | Yes   | Auto |

## 2. SSL/TLS

- **Encryption mode**: Full (strict)
- **Origin Certificates**: Generate one in Cloudflare → SSL/TLS → Origin Server
  - Copy cert → `/etc/ssl/z86.dev/origin.pem`
  - Copy key → `/etc/ssl/z86.dev/origin-key.pem`

## 3. Recommended Cloudflare Settings

- **Auto HTTPS Rewrites**: ON
- **Always Use HTTPS**: ON
- **Minimum TLS Version**: 1.2
- **Brotli**: ON
- **Browser Cache TTL**: 4 hours (static assets cached by nginx)
- **Security Level**: Medium

## 4. Page Rules (optional)

- `z86.dev/api/*` → Cache Level: Bypass
- `z86.dev/ws` → Cache Level: Bypass

## 5. Deploy

```bash
# First time
./dashboard/deploy/deploy.sh <SERVER_IP> --setup

# Updates
./dashboard/deploy/deploy.sh <SERVER_IP>
```
