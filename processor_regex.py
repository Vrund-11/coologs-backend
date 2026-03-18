import re

# Tier 1: Regex Classification (Fast and Rule-based)
def classify_with_regex(log_message: str):
    """
    Classify log messages using Regular Expressions (Regex).
    Regex is extremely fast and great for known, repeating patterns.
    """
    
    # Define rules: Pattern to match -> Category to assign
    regex_patterns = {
        # ---------- OpenStack Infrastructure ----------
        r"^nova\.osapi_compute\.wsgi\.server.*": "OpenStack API Server",
        r"^GET\s+\/v2\/.*": "HTTP Status",
        r"^POST\s+\/v2\/.*": "HTTP Status",
        r"^nova\.compute\.claims.*": "OpenStack Compute Claims",
        r"^nova\.compute\.resource_tracker.*": "OpenStack Resource Tracker",

        # ---------- Common User Activities ----------
        r"User\s+User\d+\s+logged\s+(in|out)\.?": "User Action",
        r"Account with ID\s+\w+\s+created by\s+.*": "User Action",

        # ---------- Authentication & Security Issues ----------
        r"(Multiple|Unauthorized|Invalid|Denied).*login.*": "Authentication Failure",
        r"(has escalated|privilege).*admin": "Privilege Escalation",

        # ---------- Regular System Notifications ----------
        r"Backup\s+(started|ended)\s+at\s+.*": "System Notification",
        r"Backup completed successfully\.": "System Notification",
        r"System updated to version\s+.*": "System Notification",
        r"Disk cleanup completed successfully\.": "System Notification",
        r"System reboot initiated by user\s+.*": "System Notification",
        r"File\s+.*\s+uploaded successfully by user\s+.*": "System Notification",

        # ---------- Critical Failures ----------
        r"(Replication|replication|Shard).*failed|did not complete": "Replication Failure",
        r"(Critical|Essential|Failure).*system.*component": "Critical System Failure",
        r"(RAID|disk).*failure|faulty disks": "Disk / RAID Failure",
        r"(kernel panic|boot process|boot sequence).*failed": "Kernel / Boot Failure",
        r"System configuration.*(invalid|corrupted|failure|errors)": "Configuration Error",

        # ---------- External Service Problems ----------
        r"(Email|Mail).*sending.*(issue|fault|error|glitch)": "Email Service Issue",
        r"(SSL|health check).*failed": "Service Health / SSL Issue",
        r"Module\s+X.*(invalid|mismatch|format)": "Module Input Error",

        # ---------- Real-time Security Alerts ----------
        r"(Abnormal|Anomalous|suspicious|Security alert).*server": "Security Alert"
    }

    # Loop through each rule and see if the log message matches
    for pattern, label in regex_patterns.items():
        # Using re.IGNORECASE to make the check case-insensitive
        if re.search(pattern, log_message, re.IGNORECASE):
            return label

    # Return None if no pattern matches, moving the log to the next tier
    return None

if __name__ == "__main__":
    # Test cases to verify the regex patterns work
    print("Testing User Action:", classify_with_regex("User User123 logged in."))
    print("Testing Notification:", classify_with_regex("Backup completed successfully."))
    print("Testing Service Issue:", classify_with_regex("Email service experiencing issues with sending"))
    print("Testing Unknown Log:", classify_with_regex("Hey Bro, just checking in!"))
