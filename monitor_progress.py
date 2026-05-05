import smtplib, time, os, getpass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation', 'eval_output.log')
TO_EMAIL   = 'mateisoro@gmail.com'
FROM_EMAIL = 'mateisoro@gmail.com'
INTERVAL_MINUTES = 30


def send_email(subject, body, password):
    msg = MIMEMultipart()
    msg['From'] = FROM_EMAIL; msg['To'] = TO_EMAIL; msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.starttls(); s.login(FROM_EMAIL, password)
        s.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())


def parse_progress(content):
    lines = content.strip().split('\n')
    keep = [l.strip() for l in lines[-80:]
            if any(k in l for k in [
                'Tier','Split','Subject','MLP=','GAT=','BINARY','FINAL',
                'DONE','Results saved','arousal','valence','ERROR','===','---'])]
    return '\n'.join(keep[-25:]) if keep else '\n'.join(lines[-15:])


if __name__ == '__main__':
    print(f'Log file : {LOG_FILE}')
    print(f'Recipient: {TO_EMAIL}')
    print(f'Interval : {INTERVAL_MINUTES} min')
    print()
    print('Generate an App Password at: https://myaccount.google.com/apppasswords')
    print('  (Google Account -> Security -> 2-Step Verification -> App Passwords)')
    print()
    password = getpass.getpass('Gmail App Password: ')

    try:
        send_email(
            'DeepGAT: Monitor started',
            f'Monitor started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'Watching: {LOG_FILE}\nUpdates every {INTERVAL_MINUTES} minutes.',
            password)
        print('Test email sent. Watching log...\n')
    except Exception as e:
        print(f'\nERROR: {e}')
        raise SystemExit(1)

    while True:
        time.sleep(INTERVAL_MINUTES * 60)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if not os.path.exists(LOG_FILE):
            subject = f'DeepGAT [{now}] - log not found'
            body    = f'Log file not found: {LOG_FILE}'
        else:
            with open(LOG_FILE, encoding='utf-8', errors='replace') as f:
                content = f.read()
            done    = 'DONE' in content or 'Results saved to' in content
            subject = (f'DeepGAT: EVALUATION COMPLETE! [{now}]' if done
                       else f'DeepGAT: Progress update [{now}]')
            body    = (f'Time   : {now}\nLogSize: {len(content):,} chars\n\n'
                       f'Recent output:\n{"="*40}\n{parse_progress(content)}')
        try:
            send_email(subject, body, password)
            print(f'[{now}] Email sent.')
        except Exception as e:
            print(f'[{now}] Email failed: {e}')
