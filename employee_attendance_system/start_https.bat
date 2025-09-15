@echo off
echo ===================================
echo    BHK Tech - HTTPS Server Setup
echo ===================================
echo.

cd /d "%~dp0backend"

echo Generating self-signed certificate for HTTPS...
echo.

REM Tạo certificate tự ký với OpenSSL (nếu có) hoặc sử dụng PowerShell
powershell -Command "& {
    try {
        # Tạo self-signed certificate
        $cert = New-SelfSignedCertificate -DnsName 'localhost', '*.localhost', '127.0.0.1' -CertStoreLocation 'cert:\LocalMachine\My' -KeyAlgorithm RSA -KeyLength 2048 -NotAfter (Get-Date).AddYears(1)
        
        # Export certificate
        $pwd = ConvertTo-SecureString -String 'bhktech2024' -Force -AsPlainText
        Export-PfxCertificate -Cert $cert -FilePath 'cert.pfx' -Password $pwd
        
        # Export public key
        Export-Certificate -Cert $cert -FilePath 'cert.cer'
        
        Write-Host 'Certificate created successfully!'
        Write-Host 'Files: cert.pfx, cert.cer'
        Write-Host 'Password: bhktech2024'
    } catch {
        Write-Host 'Error creating certificate: ' $_.Exception.Message
        Write-Host 'Please install OpenSSL or use manual certificate creation'
    }
}"

echo.
echo Certificate generation completed!
echo.
echo Starting HTTPS server...

python app_https.py

pause