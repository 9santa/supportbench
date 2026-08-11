Retrievers: tfidf, bm25
Split: dev
Failure cutoff: top 3

Aggregate metrics:

Retriever     Recall@1  Recall@3  Recall@5       MRR
tfidf           0.3400    0.7825    0.9600    0.5907
bm25            0.3525    0.8025    0.9525    0.6005

Per-query differences:

[BM25 BETTER] q0001
Query: как настроить vpn на ubuntu
Relevant: vpn_ubuntu_setup
TFIDF rank: 2
BM25 rank: 1

[TFIDF BETTER] q0003
Query: не работает vpn на ubuntu
Relevant: vpn_ubuntu_troubleshooting
TFIDF rank: 1
BM25 rank: 2

[ALL FAILED@3] q0004
Query: подключение зависает что проверить
Relevant: vpn_ubuntu_troubleshooting
TFIDF: software_install_recovery, vpn_ubuntu_recovery, printer_office_recovery, docker_registry_recovery, vpn_ubuntu_troubleshooting
BM25: printer_office_recovery, software_install_recovery, vpn_ubuntu_recovery, vpn_ubuntu_troubleshooting, printer_office_troubleshooting

[ALL FAILED@3] q0005
Query: как восстановить vpn на ubuntu после сбоя
Relevant: vpn_ubuntu_recovery
TFIDF: vpn_ubuntu_setup, vpn_ubuntu_troubleshooting, vpn_ubuntu_faq, shared_folder_recovery, vpn_ubuntu_recovery
BM25: vpn_ubuntu_setup, vpn_ubuntu_faq, vpn_ubuntu_troubleshooting, vpn_ubuntu_recovery, shared_folder_recovery

[ALL FAILED@3] q0006
Query: потеряна конфигурация vpn ubuntu
Relevant: vpn_ubuntu_recovery
TFIDF: vpn_ubuntu_troubleshooting, vpn_ubuntu_setup, vpn_ubuntu_faq, vpn_ubuntu_recovery, vpn_windows_setup
BM25: vpn_ubuntu_setup, vpn_ubuntu_troubleshooting, vpn_ubuntu_faq, vpn_ubuntu_recovery, vpn_windows_setup

[TFIDF BETTER] q0008
Query: что запрещено при работе с vpn ubuntu
Relevant: vpn_ubuntu_faq
TFIDF rank: 3
BM25 rank: 4

[TFIDF BETTER] q0010
Query: первичное подключение vpn windows
Relevant: vpn_windows_setup
TFIDF rank: 1
BM25 rank: 2

[TFIDF BETTER] q0011
Query: не работает vpn на windows
Relevant: vpn_windows_troubleshooting
TFIDF rank: 2
BM25 rank: 3

[TFIDF BETTER] q0012
Query: клиент сообщает об ошибке аутентификации что проверить
Relevant: vpn_windows_troubleshooting
TFIDF rank: 1
BM25 rank: 2

[BM25 BETTER] q0013
Query: как восстановить vpn на windows после сбоя
Relevant: vpn_windows_recovery
TFIDF rank: 4
BM25 rank: 3

[ALL FAILED@3] q0014
Query: потеряна конфигурация vpn windows
Relevant: vpn_windows_recovery
TFIDF: vpn_windows_setup, vpn_windows_troubleshooting, vpn_windows_faq, vpn_windows_recovery, vpn_macos_setup
BM25: vpn_windows_setup, vpn_windows_troubleshooting, vpn_windows_faq, vpn_windows_recovery, vpn_macos_setup

[BM25 BETTER] q0015
Query: правила использования vpn на windows
Relevant: vpn_windows_faq
TFIDF rank: 2
BM25 rank: 1

[BM25 BETTER] q0018
Query: первичное подключение vpn macos
Relevant: vpn_macos_setup
TFIDF rank: 3
BM25 rank: 2

[BM25 BETTER] q0021
Query: как восстановить vpn на macos после сбоя
Relevant: vpn_macos_recovery
TFIDF rank: 4
BM25 rank: 2

[ALL FAILED@3] q0022
Query: потеряна конфигурация vpn macos
Relevant: vpn_macos_recovery
TFIDF: vpn_macos_troubleshooting, vpn_macos_faq, vpn_macos_setup, vpn_macos_recovery, zoom_screen_setup
BM25: vpn_macos_troubleshooting, vpn_macos_setup, vpn_macos_faq, vpn_macos_recovery, zoom_screen_setup

[TFIDF BETTER] q0024
Query: что запрещено при работе с vpn macos
Relevant: vpn_macos_faq
TFIDF rank: 3
BM25 rank: 4

[ALL FAILED@3] q0029
Query: как восстановить корпоративный wi-fi после сбоя
Relevant: wifi_office_recovery
TFIDF: wifi_office_setup, wifi_office_faq, wifi_office_troubleshooting, wifi_office_recovery, shared_folder_recovery
BM25: wifi_office_setup, wifi_office_faq, wifi_office_troubleshooting, wifi_office_recovery, shared_folder_recovery

[BM25 BETTER] q0031
Query: правила использования корпоративный wi-fi
Relevant: wifi_office_faq
TFIDF rank: 2
BM25 rank: 1

[BM25 BETTER] q0035
Query: не работает внутренний dns
Relevant: dns_internal_troubleshooting
TFIDF rank: 4
BM25 rank: 3

[BM25 BETTER] q0037
Query: как восстановить внутренний dns после сбоя
Relevant: dns_internal_recovery
TFIDF rank: 3
BM25 rank: 1

[ALL FAILED@3] q0038
Query: потеряна конфигурация dns nslookup
Relevant: dns_internal_recovery
TFIDF: dns_internal_setup, dns_internal_troubleshooting, dns_internal_faq, dns_internal_recovery, vpn_ubuntu_troubleshooting
BM25: dns_internal_setup, dns_internal_troubleshooting, dns_internal_faq, dns_internal_recovery, vpn_ubuntu_troubleshooting

[BM25 BETTER] q0045
Query: как восстановить корпоративный proxy после сбоя
Relevant: proxy_browser_recovery
TFIDF rank: 3
BM25 rank: 1

[ALL FAILED@3] q0048
Query: что запрещено при работе с proxy pac
Relevant: proxy_browser_faq
TFIDF: proxy_browser_setup, proxy_browser_troubleshooting, proxy_browser_recovery, proxy_browser_faq, antivirus_alert_faq
BM25: proxy_browser_recovery, proxy_browser_setup, proxy_browser_troubleshooting, proxy_browser_faq, password_reset_recovery

[TFIDF BETTER] q0052
Query: одноразовый код не принимается что проверить
Relevant: gitlab_2fa_troubleshooting
TFIDF rank: 1
BM25 rank: 2

[ALL FAILED@3] q0056
Query: что запрещено при работе с gitlab 2fa
Relevant: gitlab_2fa_faq
TFIDF: gitlab_2fa_setup, gitlab_2fa_recovery, gitlab_2fa_troubleshooting, antivirus_alert_faq, gitlab_2fa_faq
BM25: gitlab_2fa_setup, gitlab_2fa_recovery, gitlab_2fa_troubleshooting, password_reset_recovery, gitlab_2fa_faq

[BM25 BETTER] q0061
Query: как восстановить сброс корпоративного пароля после сбоя
Relevant: password_reset_recovery
TFIDF rank: 4
BM25 rank: 3

[ALL FAILED@3] q0062
Query: потеряна конфигурация пароль сброс
Relevant: password_reset_recovery
TFIDF: password_reset_troubleshooting, password_reset_setup, password_reset_faq, password_reset_recovery, vpn_macos_setup
BM25: password_reset_troubleshooting, password_reset_setup, password_reset_faq, password_reset_recovery, vpn_macos_setup

[TFIDF BETTER] q0064
Query: что запрещено при работе с пароль сброс
Relevant: password_reset_faq
TFIDF rank: 3
BM25 rank: 4

[BM25 BETTER] q0065
Query: как настроить ssh-ключи
Relevant: ssh_keys_setup
TFIDF rank: 3
BM25 rank: 2

[TFIDF BETTER] q0067
Query: не работает ssh-ключи
Relevant: ssh_keys_troubleshooting
TFIDF rank: 2
BM25 rank: 3

[BM25 BETTER] q0069
Query: как восстановить ssh-ключи после сбоя
Relevant: ssh_keys_recovery
TFIDF rank: 4
BM25 rank: 2

[ALL FAILED@3] q0072
Query: что запрещено при работе с ssh ключ
Relevant: ssh_keys_faq
TFIDF: ssh_keys_setup, ssh_keys_recovery, ssh_keys_troubleshooting, ssh_keys_faq, antivirus_alert_faq
BM25: ssh_keys_recovery, ssh_keys_setup, ssh_keys_troubleshooting, ssh_keys_faq, bitlocker_recovery_recovery

[BM25 BETTER] q0073
Query: как настроить корневой сертификат компании
Relevant: corporate_ca_setup
TFIDF rank: 3
BM25 rank: 2

[BM25 BETTER] q0074
Query: первичное подключение сертификат ca
Relevant: corporate_ca_setup
TFIDF rank: 2
BM25 rank: 1

[TFIDF BETTER] q0075
Query: не работает корневой сертификат компании
Relevant: corporate_ca_troubleshooting
TFIDF rank: 2
BM25 rank: 3

[BM25 BETTER] q0077
Query: как восстановить корневой сертификат компании после сбоя
Relevant: corporate_ca_recovery
TFIDF rank: 3
BM25 rank: 2

[ALL FAILED@3] q0078
Query: потеряна конфигурация сертификат ca
Relevant: corporate_ca_recovery
TFIDF: corporate_ca_troubleshooting, corporate_ca_setup, docker_registry_troubleshooting, docker_registry_recovery, docker_registry_faq
BM25: corporate_ca_setup, corporate_ca_troubleshooting, docker_registry_troubleshooting, docker_registry_faq, docker_registry_recovery

[BM25 BETTER] q0085
Query: как восстановить профиль outlook после сбоя
Relevant: outlook_profile_recovery
TFIDF rank: 2
BM25 rank: 1

[TFIDF BETTER] q0086
Query: потеряна конфигурация outlook почта
Relevant: outlook_profile_recovery
TFIDF rank: 3
BM25 rank: 4

[BM25 BETTER] q0087
Query: правила использования профиль outlook
Relevant: outlook_profile_faq
TFIDF rank: 3
BM25 rank: 1

[ALL FAILED@3] q0088
Query: что запрещено при работе с outlook почта
Relevant: outlook_profile_faq
TFIDF: outlook_profile_setup, outlook_profile_troubleshooting, outlook_profile_recovery, outlook_profile_faq, antivirus_alert_faq
BM25: outlook_profile_setup, outlook_profile_troubleshooting, outlook_profile_recovery, mailbox_quota_setup, mailbox_quota_troubleshooting

[ALL FAILED@3] q0094
Query: потеряна конфигурация почта квота
Relevant: mailbox_quota_recovery
TFIDF: mailbox_quota_setup, mailbox_quota_troubleshooting, mailbox_quota_faq, outlook_profile_setup, outlook_profile_troubleshooting
BM25: mailbox_quota_setup, mailbox_quota_troubleshooting, mailbox_quota_faq, outlook_profile_setup, outlook_profile_troubleshooting

[TFIDF BETTER] q0096
Query: что запрещено при работе с почта квота
Relevant: mailbox_quota_faq
TFIDF rank: 3
BM25 rank: 4

[TFIDF BETTER] q0100
Query: собеседники не слышат пользователя что проверить
Relevant: teams_audio_setup, teams_audio_troubleshooting
TFIDF rank: 2
BM25 rank: 3

[BM25 BETTER] q0101
Query: как восстановить звук в microsoft teams после сбоя
Relevant: teams_audio_recovery
TFIDF rank: 4
BM25 rank: 3

[ALL FAILED@3] q0104
Query: что запрещено при работе с teams микрофон
Relevant: teams_audio_faq
TFIDF: teams_audio_setup, teams_audio_recovery, teams_audio_troubleshooting, antivirus_alert_faq, teams_audio_faq
BM25: teams_audio_setup, teams_audio_recovery, teams_audio_troubleshooting, password_reset_recovery, teams_audio_faq

[ALL FAILED@3] q0107
Query: не работает демонстрация экрана в zoom
Relevant: zoom_screen_troubleshooting
TFIDF: zoom_screen_setup, zoom_screen_faq, zoom_screen_recovery, zoom_screen_troubleshooting, browser_cache_recovery
BM25: zoom_screen_faq, zoom_screen_setup, zoom_screen_recovery, zoom_screen_troubleshooting, browser_cache_troubleshooting

[BM25 BETTER] q0111
Query: правила использования демонстрация экрана в zoom
Relevant: zoom_screen_faq
TFIDF rank: 2
BM25 rank: 1

[BM25 BETTER] q0112
Query: что запрещено при работе с zoom экран
Relevant: zoom_screen_faq
TFIDF rank: 4
BM25 rank: 1

[BM25 BETTER] q0115
Query: не работает офисный принтер
Relevant: printer_office_troubleshooting
TFIDF rank: 4
BM25 rank: 3

[BM25 BETTER] q0117
Query: как восстановить офисный принтер после сбоя
Relevant: printer_office_recovery
TFIDF rank: 2
BM25 rank: 1

[TFIDF BETTER] q0118
Query: потеряна конфигурация принтер печать
Relevant: printer_office_recovery
TFIDF rank: 3
BM25 rank: 4

[ALL FAILED@3] q0123
Query: не работает восстановление bitlocker
Relevant: bitlocker_recovery_troubleshooting
TFIDF: bitlocker_recovery_setup, bitlocker_recovery_recovery, bitlocker_recovery_faq, bitlocker_recovery_troubleshooting, browser_cache_recovery
BM25: bitlocker_recovery_setup, bitlocker_recovery_recovery, bitlocker_recovery_faq, bitlocker_recovery_troubleshooting, browser_cache_recovery

[ALL FAILED@3] q0128
Query: что запрещено при работе с bitlocker recovery key
Relevant: bitlocker_recovery_faq
TFIDF: bitlocker_recovery_setup, bitlocker_recovery_recovery, bitlocker_recovery_troubleshooting, bitlocker_recovery_faq, gitlab_2fa_recovery
BM25: bitlocker_recovery_setup, bitlocker_recovery_recovery, bitlocker_recovery_troubleshooting, gitlab_2fa_recovery, gitlab_2fa_setup

[ALL FAILED@3] q0134
Query: потеряна конфигурация антивирус malware
Relevant: antivirus_alert_recovery
TFIDF: antivirus_alert_troubleshooting, antivirus_alert_setup, antivirus_alert_faq, antivirus_alert_recovery
BM25: antivirus_alert_troubleshooting, antivirus_alert_setup, antivirus_alert_faq, antivirus_alert_recovery

[TFIDF BETTER] q0140
Query: установка зависает что проверить
Relevant: software_install_setup, software_install_troubleshooting
TFIDF rank: 1
BM25 rank: 2

[BM25 BETTER] q0141
Query: как восстановить установка корпоративного по после сбоя
Relevant: software_install_recovery
TFIDF rank: 3
BM25 rank: 2

[ALL FAILED@3] q0144
Query: что запрещено при работе с установка software center
Relevant: software_install_faq
TFIDF: software_install_setup, software_install_troubleshooting, software_install_recovery, software_install_faq, antivirus_alert_faq
BM25: software_install_setup, software_install_recovery, software_install_troubleshooting, software_install_faq, password_reset_recovery

[BM25 BETTER] q0145
Query: как настроить запрос прав доступа
Relevant: access_request_setup
TFIDF rank: 3
BM25 rank: 2

[TFIDF BETTER] q0147
Query: не работает запрос прав доступа
Relevant: access_request_troubleshooting
TFIDF rank: 2
BM25 rank: 3

[TFIDF BETTER] q0148
Query: заявка согласована что проверить
Relevant: access_request_troubleshooting
TFIDF rank: 1
BM25 rank: 2

[TFIDF BETTER] q0149
Query: как восстановить запрос прав доступа после сбоя
Relevant: access_request_recovery
TFIDF rank: 2
BM25 rank: 3

[ALL FAILED@3] q0152
Query: что запрещено при работе с доступ роль
Relevant: access_request_faq
TFIDF: access_request_setup, access_request_troubleshooting, antivirus_alert_faq, access_request_recovery, kubernetes_access_recovery
BM25: access_request_setup, access_request_recovery, access_request_troubleshooting, kubernetes_access_recovery, password_reset_recovery

[TFIDF BETTER] q0156
Query: папка не открывается что проверить
Relevant: shared_folder_troubleshooting
TFIDF rank: 1
BM25 rank: 2

[ALL FAILED@3] q0168
Query: что запрещено при работе с rdp remote desktop
Relevant: remote_desktop_faq
TFIDF: remote_desktop_setup, remote_desktop_troubleshooting, remote_desktop_recovery, remote_desktop_faq, antivirus_alert_faq
BM25: remote_desktop_troubleshooting, remote_desktop_setup, remote_desktop_recovery, remote_desktop_faq, password_reset_recovery

[TFIDF BETTER] q0172
Query: файл удалён что проверить
Relevant: backup_restore_troubleshooting
TFIDF rank: 1
BM25 rank: 2

[ALL FAILED@3] q0174
Query: потеряна конфигурация backup резервная копия
Relevant: backup_restore_recovery
TFIDF: backup_restore_setup, backup_restore_troubleshooting
BM25: backup_restore_setup, backup_restore_troubleshooting

[ALL FAILED@3] q0176
Query: что запрещено при работе с backup резервная копия
Relevant: backup_restore_faq
TFIDF: backup_restore_setup, backup_restore_troubleshooting, antivirus_alert_faq, password_reset_recovery, mailbox_quota_recovery
BM25: backup_restore_setup, backup_restore_troubleshooting, password_reset_recovery, mailbox_quota_recovery, antivirus_alert_faq

[TFIDF BETTER] q0177
Query: как настроить кэш и cookies браузера
Relevant: browser_cache_setup
TFIDF rank: 1
BM25 rank: 2

[BM25 BETTER] q0181
Query: как восстановить кэш и cookies браузера после сбоя
Relevant: browser_cache_recovery
TFIDF rank: 4
BM25 rank: 3

[ALL FAILED@3] q0182
Query: потеряна конфигурация браузер cache
Relevant: browser_cache_recovery
TFIDF: browser_cache_setup, browser_cache_troubleshooting, proxy_browser_troubleshooting, browser_cache_recovery, browser_cache_faq
BM25: browser_cache_setup, browser_cache_troubleshooting, proxy_browser_troubleshooting, browser_cache_faq, browser_cache_recovery

[ALL FAILED@3] q0184
Query: что запрещено при работе с браузер cache
Relevant: browser_cache_faq
TFIDF: browser_cache_setup, browser_cache_troubleshooting, antivirus_alert_faq, browser_cache_recovery, proxy_browser_troubleshooting
BM25: browser_cache_setup, browser_cache_troubleshooting, browser_cache_recovery, proxy_browser_recovery, corporate_ca_recovery

[TFIDF BETTER] q0185
Query: как настроить доступ к docker registry
Relevant: docker_registry_setup
TFIDF rank: 1
BM25 rank: 2

[TFIDF BETTER] q0189
Query: как восстановить доступ к docker registry после сбоя
Relevant: docker_registry_recovery
TFIDF rank: 2
BM25 rank: 3

[BM25 BETTER] q0191
Query: правила использования доступ к docker registry
Relevant: docker_registry_faq
TFIDF rank: 2
BM25 rank: 1

[BM25 BETTER] q0192
Query: что запрещено при работе с docker registry
Relevant: docker_registry_faq
TFIDF rank: 4
BM25 rank: 3

[BM25 BETTER] q0197
Query: как восстановить доступ к kubernetes после сбоя
Relevant: kubernetes_access_recovery
TFIDF rank: 4
BM25 rank: 2

[ALL FAILED@3] q0198
Query: потеряна конфигурация kubernetes kubectl
Relevant: kubernetes_access_recovery
TFIDF: kubernetes_access_setup, kubernetes_access_troubleshooting, kubernetes_access_faq, kubernetes_access_recovery
BM25: kubernetes_access_setup, kubernetes_access_troubleshooting, kubernetes_access_faq, kubernetes_access_recovery

Comparison summary:
Queries: 200
All succeeded@3: 156
Mixed results@3: 17
All failed@3: 27
TFIDF better by RR in: 26 queries
BM25 better by RR in: 30 queries
Tied by RR: 144


# 11 из 27 общих провалов top-3 — шаблон «потеряна конфигурация → recovery»;
ещё 11 — «что запрещено при работе → FAQ».

То есть 22 из 27 общих провалов, около 81%, относятся всего к двум систематическим классам ошибок. Это отличный результат для error analysis: проблема локализована.

# Поскольку lexical retrievers достигают около 95% Recall@5, но только около 35% Recall@1, reranker должен улучшить выбор между setup, troubleshooting, recovery и FAQ-документами одной темы.


# Corpus statistics
Documents: 100
Vocabulary size: 750
Average document length: 58.96

Top terms by document frequency:
1. и, df=100
2. проверьте, df=94
3. или, df=74
4. не, df=67
5. после, df=57
6. для, df=55
7. в, df=51
8. проверки, df=50
9. термины, df=50
10. время, df=40
11. на, df=40
12. используйте, df=36
13. к, df=36
14. настройки, df=36
15. восстановление, df=31
16. vpn, df=30
17. восстановления, df=30
18. выполните, df=30
19. пароля, df=30
20. без, df=29

Document lengths:
  min:    44
  median: 60.00
  mean:   58.96
  p90:    69.00
  max:    73
  std:    7.86
  CV:     0.1333

Posting frequencies:
  postings: 5012
  tf=1:     86.97%
  tf=2:     9.24%
  tf>=3:   3.79%
  mean tf:  1.18
  max tf:   6

Тип	Документов	Средняя длина	Median	Min–Max
setup	25	47.88	48	44–53
troubleshooting	25	56.96	56	53–63
faq	25	63.12	62	59–70
recovery	25	67.88	68	64–73

# Гипотеза: почему tfidf и bm25 показывают почти одинаковые результаты?
TF-IDF и BM25 показывают близкие результаты, поскольку корпус состоит из коротких документов с низким разбросом длины и преимущественно единичными вхождениями терминов. В этих условиях механизмы BM25 — насыщение TF и нормализация длины — оказывают слабое влияние, а ранжирование обоих методов в основном определяется IDF и лексическим пересечением.


Parameter: b
Configurations: 5
Output: /home/cohle/machine_learning/supportbench/results/bm25_ablation/b

Experiment         k1      b      R@1      R@3      R@5      MRR
b_0_00           1.50   0.00   0.3250   0.7800   0.9400   0.5806
b_0_25           1.50   0.25   0.3425   0.8075   0.9400   0.5989
b_0_50           1.50   0.50   0.3425   0.8025   0.9450   0.5978
b_0_75           1.50   0.75   0.3525   0.8025   0.9525   0.6005
b_1_00           1.50   1.00   0.3925   0.7950   0.9475   0.6193
(.venv) ➜  supportbench git:(master) ✗ python -m scripts.run_bm25_ablation --parameter k1
Parameter: k1
Configurations: 7
Output: /home/cohle/machine_learning/supportbench/results/bm25_ablation/k1

Experiment         k1      b      R@1      R@3      R@5      MRR
k1_0_2           0.20   0.75   0.3975   0.7875   0.9300   0.6352
k1_0_5           0.50   0.75   0.4025   0.7925   0.9375   0.6362
k1_0_8           0.80   0.75   0.3975   0.7975   0.9400   0.6298
k1_1_2           1.20   0.75   0.3525   0.8025   0.9425   0.6002
k1_1_5           1.50   0.75   0.3525   0.8025   0.9525   0.6005
k1_2_0           2.00   0.75   0.3625   0.8000   0.9600   0.6045
k1_3_0           3.00   0.75   0.3525   0.7800   0.9600   0.5964


Результаты частично опровергают первоначальную гипотезу. Параметры BM25 влияют заметно на результ.
Причины:
1) Длина систематически кодирует тип документа, так как есть зависимость между типом и длиной.
2) Важные тематические слова повторяются в заголовке, тексте и списках терминов.

Лучший результат зафиксирован при комбинации k1 = 0.5, b = 1.0:
Retriever: bm25
Split: dev
Queries: 200

Recall@1: 0.4300
Recall@3: 0.7925
Recall@5: 0.9375
MRR:      0.6488

BM25 default: k1=1.5, b=0.75
BM25 tuned:   k1=0.5, b=1.0
Metric	Default	 Tuned	Absolute Change	% Change
Recall@1	0.3525	0.4300	+0.0775	+21.99%
Recall@3	0.8025	0.7925	−0.0100	−1.25%
Recall@5	0.9525	0.9375	−0.0150	−1.57%
MRR	0.6005	0.6488	+0.0483	+8.04%

### Summary:

Large improvement in top-1 recall (+22%) and a solid gain in MRR (+8%).
Slight drops in Recall@3 and Recall@5 (−1.25% and −1.57%).





# Flow:
query
  ↓
encode_queries([query])
  ↓
матрица shape (1, dimension)
  ↓
берём embeddings[0]
  ↓
vector index search
  ↓
VectorSearchResult
  ↓
SearchResult с rank 1, 2, 3...




# First evaluation of DenseRetriever
dense-dev-v1

model: intfloat/multilingual-e5-base
index: FAISS IndexFlatIP
normalized: true
document_format: title_newline_text

Retriever: dense
Split: dev
Queries: 200

Recall@1: 0.7600
Recall@3: 0.9600
Recall@5: 0.9875
Recall@10: 0.9875
MRR:      0.8842



# Hybrid Weighted RRF
### bm25 weight = 1, dense weight = 1
Retriever: hybrid
Split: dev
Queries: 200

Recall@1: 0.6325
Recall@3: 0.9125
Recall@5: 0.9800
Recall@10: 0.9800
MRR:      0.7971

### bm25 weight = 1, dense weight = 2
Retriever: hybrid
Split: dev
Queries: 200

Recall@1: 0.7075
Recall@3: 0.9300
Recall@5: 0.9825
Recall@10: 0.9825
MRR:      0.8471



### bm25 weight = 1, dense weight = 3
Retriever: hybrid
Split: dev
Queries: 200

Recall@1: 0.7350
Recall@3: 0.9450
Recall@5: 0.9825
Recall@10: 0.9825
MRR:      0.8658


Конфигурация	R@1	R@3	R@5	R@10	MRR
Dense	0.7600	0.9600	0.9875	0.9875	0.8842
RRF 1:1	0.6325	0.9125	0.9800	0.9800	0.7971
RRF 1:2	0.7075	0.9300	0.9825	0.9825	0.8471
RRF 1:3	0.7350	0.9450	0.9825	0.9825	0.8658

Потери относительно dense:

Конфигурация	Δ R@1	Δ R@3	Δ R@5	Δ MRR
RRF 1:1	−0.1275	−0.0475	−0.0075	−0.0871
RRF 1:2	−0.0525	−0.0300	−0.0050	−0.0371
RRF 1:3	−0.0250	−0.0150	−0.0050	−0.0184


rrf_k = 20, RRF 1:3
Recall@1: 0.7400
Recall@3: 0.9475
Recall@5: 0.9875
Recall@10: 0.9875
MRR:      0.8702


rrf_k = 10, RRF 1:3
Recall@1: 0.7450
Recall@3: 0.9500
Recall@5: 0.9900
Recall@10: 0.9900
MRR:      0.8742


rrf_k = 20, RRF 1 : 5
Recall@1: 0.7550
Recall@3: 0.9500
Recall@5: 0.9900
Recall@10: 0.9900
MRR:      0.8796




# FIRST RESULTS ON NEW DOCUMENTS CORPUS "documents_v2.jsonl, OLD QUERIES "queries_dev.jsonl"
### DENSE RETRIEVER
Retriever: dense
Split: dev
Queries: 200

Recall@1: 0.7200
Recall@3: 0.9075
Recall@5: 0.9450
Recall@10: 0.9675
MRR:      0.8420



### BM25 RETRIEVER
Retriever: bm25
Split: dev
Queries: 200

Recall@1: 0.3375
Recall@3: 0.7175
Recall@5: 0.8425
Recall@10: 0.8800
MRR:      0.5556


### HYBRID RETRIEVER (DEFAULT CONFIG)
Retriever: hybrid
Split: dev
Queries: 200

Recall@1: 0.6200
Recall@3: 0.8425
Recall@5: 0.9225
Recall@10: 0.9275
MRR:      0.7635



# FIRST RESULTS ON NEW DOCUMENTS CORPUS "documents_v2.jsonl, NEW QUERIES "queries_v2_{dev/frozen_test}.jsonl"
### DENSE
Retriever: dense
Split: dev
Queries: 500

Recall@1: 0.6293
Recall@3: 0.8343
Recall@5: 0.8827
Recall@10: 0.9263
MRR:      0.7827



### BM25
Retriever: bm25
Split: dev
Queries: 500

Recall@1: 0.3780
Recall@3: 0.6813
Recall@5: 0.8263
Recall@10: 0.8693
MRR:      0.5770



### HYBRID
Retriever: hybrid
Split: dev
Queries: 500

Recall@1: 0.5690
Recall@3: 0.7953
Recall@5: 0.8883
Recall@10: 0.9157
MRR:      0.7380




# GRID SEARCH ON WEIGHTED RRF (params: dense retriever weight and rrf_k)

















# Reranker Comparison
Reranker comparison

Queries: 500
Reranker candidate pool: 20
Final result count: 10

Candidate source metrics:

Source                    R@1      R@3      R@5     R@10     R@20     R@50      MRR
dense                  0.6488   0.8601   0.9100   0.9550   0.9735   0.9735   0.8069
rrf_standalone         0.6619   0.8581   0.9268   0.9636   0.9821   0.9821   0.8178
rrf_candidate          0.6100   0.8416   0.9227   0.9660   0.9835   0.9835   0.7852

After cross-encoder reranking:

Source                     R@1       R@3       R@5      R@10       MRR
dense                   0.7684    0.8808    0.9244    0.9577    0.8852
rrf_standalone          0.7649    0.8804    0.9241    0.9625    0.8841
rrf_candidate           0.7670    0.8825    0.9261    0.9629    0.8856

Reranker deltas against each source:

Source                    ΔR@1      ΔR@3      ΔR@5     ΔR@10      ΔMRR
dense                   0.1196    0.0206    0.0144    0.0027    0.0783
rrf_standalone          0.1031    0.0223   -0.0027   -0.0010    0.0663
rrf_candidate           0.1570    0.0409    0.0034   -0.0031    0.1004













# Performance benchmark and metrics
reranking latency:
    time spent inside reranker.rerank()

total latency:
    candidate retrieval
    + document formatting
    + reranking
    + final result construction

VRAM:
    peak allocated
    peak reserved
    incremental allocation during reranking

batch throughput:
    query-document pairs / reranking second
    effective batches / reranking second


Performance benchmark:

GPU: NVIDIA GeForce RTX 3070 Ti
GPU power limit: 150 W
Manual GPU pauses: none

Source                 Retr p50  Rerank p50  Rerank p95  Total p50  Total p95    Pairs/s
dense                      6.47      581.98      874.77     588.91     881.70      30.85
rrf_standalone             8.68      543.40      888.48     553.03     898.72      30.88
rrf_candidate              8.67      562.53      901.00     571.00     909.94      30.65

VRAM:

Source                Peak alloc GiB  Peak reserve GiB    Rerank +GiB   Batches/s
dense                          3.507             4.348          0.348        3.08
rrf_standalone                 3.507             4.348          0.348        3.09
rrf_candidate                  3.507             4.348          0.348        3.06





# RAG Pipeline
BM25 + Dense
      ↓
candidate RRF
      ↓
cross-encoder reranker
      ↓ SearchResult
RetrievalPipeline
      ↓ RetrievedDocument
ContextBuilder
      ↓ RAGContext





# LLM Answer (gemma3:4b)

Query: "потерял телефон с кодами gitlab"

Decision: answer
Answer:
Если вы потеряли телефон с кодами gitlab, сначала проверьте точное время на телефоне и используйте один из резервных recovery codes. Если проблема сохраняется, приложите точный текст ошибки и время возникновения.
Citations:
- gitlab_2fa_troubleshooting
- gitlab_2fa_recovery
- gitlab_2fa_faq

Context:
[DOCUMENT]
doc_id: gitlab_2fa_troubleshooting
title: Устранение неполадок: Двухфакторная аутентификация GitLab
category: access
content:
Диагностика типичных ошибок и последовательность безопасных проверок. Типичная ситуация: Одноразовый код не принимается, телефон потерян или приложение-аутентификатор недоступно. Рекомендуемая диагностика: Проверьте точное время на телефоне и используйте один из резервных recovery codes. Если проблема сохраняется, приложите точный текст ошибки и время возникновения. Связанные термины: gitlab, 2fa, totp, recovery, код.
[/DOCUMENT]

[DOCUMENT]
doc_id: gitlab_2fa_recovery
title: Восстановление: Двухфакторная аутентификация GitLab
category: access
content:
Что делать после сбоя, смены устройства, пароля или конфигурации. Сначала зафиксируйте симптомы и не удаляйте рабочую конфигурацию без необходимости. Основная процедура: Проверьте точное время на телефоне и используйте один из резервных recovery codes. После восстановления повторите настройку при необходимости: Откройте настройки профиля GitLab, включите 2FA, отсканируйте QR-код и сохраните recovery codes. Проблемный сценарий: Одноразовый код не принимается, телефон потерян или приложение-аутентификатор недоступно.
[/DOCUMENT]

[DOCUMENT]
doc_id: gitlab_2fa_setup
title: Настройка: Двухфакторная аутентификация GitLab
category: access
content:
Пошаговая инструкция для первичного подключения и проверки результата. Откройте настройки профиля GitLab, включите 2FA, отсканируйте QR-код и сохраните recovery codes. После настройки выполните базовую проверку: Проверьте точное время на телефоне и используйте один из резервных recovery codes. Ключевые термины: gitlab, 2fa, totp, recovery, код.
[/DOCUMENT]

[DOCUMENT]
doc_id: gitlab_2fa_faq
title: Правила и частые вопросы: Двухфакторная аутентификация GitLab
category: access
content:
Ограничения, требования безопасности и ответы на распространённые вопросы. Резервные коды должны храниться отдельно от рабочего ноутбука. Частая ошибка пользователя: Одноразовый код не принимается, телефон потерян или приложение-аутентификатор недоступно. Для проверки используйте следующий порядок: Проверьте точное время на телефоне и используйте один из резервных recovery codes. Материал относится к теме Двухфакторная аутентификация GitLab.
[/DOCUMENT]

[DOCUMENT]
doc_id: access_edinyy_vhod_sso_gitlab_enterprise_recovery
title: Восстановление: единый вход SSO — GitLab Enterprise
category: access
content:
Тема: единый вход SSO в контексте «GitLab Enterprise». Внутренний код примера: AUTH-7420/6401. Процедура восстановления после сбоя, ошибочного изменения или замены компонента. Перед началом сохраните текущее состояние и подтвердите владельца объекта. Основные действия: отзовите активные сессии и повторите регистрацию MFA; выпустите временный recovery code с ограниченным сроком. После восстановления проверьте совпадение UPN, email и идентификатора сотрудника и статус учётной записи, группы и дату истечения. Если снова возникает сценарий «вход блокируется после смены устройства или номера телефона», приложите код AUTH-7420/6401 и остановите повторные изменения до анализа причины. Для поиска также используйте англоязычные формулировки: health check, rollback, permission denied, stale cache, owner approval.
[/DOCUMENT]

Raw response:
{"decision": "answer", "answer": "Если вы потеряли телефон с кодами gitlab, сначала проверьте точное время на телефоне и используйте один из резервных recovery codes. Если проблема сохраняется, приложите точный текст ошибки и время возникновения.", "citation_ids": ["gitlab_2fa_troubleshooting", "gitlab_2fa_recovery", "gitlab_2fa_faq"]}


### ПРОБЛЕМА
Основная проблема в самом корпусе документов, они слабые, неточные, плохо сгенерированные. Сам пайплайн и модель работают корректно, но информацию которую модель берет из документов фактически слабая.

Вопрос в том, где мне взять этот хороший корпус на допустим 5000 документов. То что сейчас - это сгенерированное LLM, и видимо на таком обьеме (5000 документов) он неспособен сгенироровать их качественно, как если бы запрос был допустим 20 документов.


## РЕШЕНИЕ
Публичных датасетов, точно соответствующих внутренней IT-поддержке конкретной компании, практически нет, потому что реальные базы знаний и политики закрыты. Поэтому для проекта использован открытый TechQA — технический support-корпус IBM. Архитектура ingestion и retrieval не привязана к IBM и рассчитана на замену источника корпоративной базой знаний.

### Следствия и ограничения
1. Корпус смещён в сторону продуктов и терминологии IBM.
2. Technotes зафиксированы на срезе 2019 года и могут содержать
   устаревшие процедуры.
3. TechQA имеет сравнительно немного размеченных QA-пар.
4. Результаты нельзя автоматически переносить на любую
   корпоративную базу знаний без domain adaptation.



Building BM25 index for 28,481 documents...
Index built: 28,481 documents, 235,124 terms, avg length 537.0 tokens

Split: dev
Queries: 310 total, 160 labeled, 150 unlabeled
Recall@1:  0.4188
Recall@3:  0.5687
Recall@5:  0.6125
Recall@10: 0.6813
Recall@20: 0.7375
Recall@50: 0.8125
MRR@10:    0.4991


Dense Index
Documents: 28,481
Embedding dimension: 768
Encoding time: 325.88s
Index build time: 0.20s
Index saved to: /home/cohle/machine_learning/supportbench/artifacts/indexes/nvidia_techqa/multilingual_e5_base


Retriever: dense
Documents: 28,481
Queries: 310
Split: dev

Queries: 310 total, 160 labeled, 150 unlabeled
Recall@1:  0.3688
Recall@3:  0.5625
Recall@5:  0.6000
Recall@10: 0.6875
Recall@20: 0.7438
Recall@50: 0.8500
MRR@10:    0.4784


Retriever: dense
Documents: 28,481
Queries: 600
Split: train

Queries: 600 total, 450 labeled, 150 unlabeled
Recall@1:  0.4067
Recall@3:  0.5244
Recall@5:  0.5622
Recall@10: 0.6289
Recall@20: 0.6956
Recall@50: 0.7778
MRR@10:    0.4766


Retriever: hybrid
Documents: 28,481
Queries: 310
Split: dev

Queries: 310 total, 160 labeled, 150 unlabeled
Recall@1:  0.4437
Recall@3:  0.5938
Recall@5:  0.6375
Recall@10: 0.7312
Recall@20: 0.8063
Recall@50: 0.8875
MRR@10:    0.5373



## Grid search on hybrid retriever
dense_weight:
1.0, 1.5, 2.0, 3.0

rrf_k:
10, 20, 40, 60

source candidate_k:
100



| dense_weight | rrf_k | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | MRR@10 |
| -----------: | ----: | -------: | -------: | -------: | --------: | --------: | --------: | -----: |
|          1.0 |    10 |   0.4578 |   0.5644 |   0.6111 |    0.6756 |    0.7511 |    0.8311 | 0.5251 |
|          1.5 |    10 |   0.4600 |   0.5578 |   0.6111 |    0.6778 |    0.7444 |    0.8200 | 0.5249 |
|          2.0 |    10 |   0.4533 |   0.5444 |   0.5956 |    0.6800 |    0.7378 |    0.8156 | 0.5169 |
|          3.0 |    10 |   0.4400 |   0.5489 |   0.5800 |    0.6733 |    0.7244 |    0.8067 | 0.5071 |
|          1.0 |    20 |   0.4556 |   0.5578 |   0.6044 |    0.6800 |    0.7489 |    0.8356 | 0.5228 |
|          1.5 |    20 |   0.4600 |   0.5556 |   0.6067 |    0.6844 |    0.7311 |    0.8222 | 0.5250 |
|          2.0 |    20 |   0.4533 |   0.5533 |   0.5933 |    0.6822 |    0.7289 |    0.8244 | 0.5199 |
|          3.0 |    20 |   0.4422 |   0.5444 |   0.5822 |    0.6733 |    0.7244 |    0.7911 | 0.5084 |
|          1.0 |    40 |   0.4533 |   0.5600 |   0.5978 |    0.6644 |    0.7511 |    0.8289 | 0.5202 |
|          1.5 |    40 |   0.4622 |   0.5600 |   0.5978 |    0.6733 |    0.7311 |    0.8244 | 0.5246 |
|          2.0 |    40 |   0.4533 |   0.5511 |   0.5933 |    0.6733 |    0.7289 |    0.7956 | 0.5195 |
|          3.0 |    40 |   0.4489 |   0.5467 |   0.5844 |    0.6756 |    0.7267 |    0.7867 | 0.5136 |
|          1.0 |    60 |   0.4511 |   0.5578 |   0.5978 |    0.6600 |    0.7422 |    0.8289 | 0.5164 |
|          1.5 |    60 |   0.4622 |   0.5556 |   0.5933 |    0.6733 |    0.7311 |    0.8133 | 0.5237 |
|          2.0 |    60 |   0.4533 |   0.5511 |   0.5889 |    0.6733 |    0.7333 |    0.7889 | 0.5195 |
|          3.0 |    60 |   0.4489 |   0.5489 |   0.5889 |    0.6711 |    0.7267 |    0.7911 | 0.5153 |


## Запуск лучшей конфигурации на dev
  --retriever hybrid \
  --split dev \
  --bm25-weight 1.0 \
  --dense-weight 1.0 \
  --candidate-k 100 \
  --rrf-k 20

Retriever: hybrid
Documents: 28,481
Queries: 310
Split: dev

Queries: 310 total, 160 labeled, 150 unlabeled
Recall@1:  0.4625
Recall@3:  0.5938
Recall@5:  0.6625
Recall@10: 0.7125
Recall@20: 0.8125
Recall@50: 0.8938
MRR@10:    0.5452


## Dense Index build on chunked document corpus
Documents: 135,235
Embedding dimension: 768
Encoding time: 1062.67s
Index build time: 0.32s


Documents: 96,134
Embedding dimension: 768
Encoding time: 1049.10s
Index build time: 0.28s




| Chunk config | Retriever | Documents | BM25 weight | Dense weight | Candidate k | RRF k | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | MRR@10 |
| ------------ | --------- | --------: | ----------: | -----------: | ----------: | ----: | -------: | -------: | -------: | --------: | --------: | --------: | -----: |
| ft256o32     | bm25      |   135,235 |           — |            — |           — |     — |   0.4244 |   0.5133 |   0.5600 |    0.6178 |    0.6800 |    0.7422 | 0.4828 |
| ft384o64     | bm25      |    96,134 |           — |            — |           — |     — |   0.4089 |   0.5089 |   0.5578 |    0.6156 |    0.6733 |    0.7311 | 0.4723 |
| ft256o32     | dense     |   135,235 |           — |            — |           — |     — |   0.4333 |   0.5444 |   0.5956 |    0.6622 |    0.7244 |    0.7822 | 0.5040 |
| ft384o64     | dense     |    96,134 |           — |            — |           — |     — |   0.4133 |   0.5422 |   0.5911 |    0.6533 |    0.7244 |    0.7867 | 0.4871 |
| ft256o32     | hybrid    |   135,235 |         1.0 |          1.0 |         100 |    20 |   0.4467 |   0.5733 |   0.6178 |    0.6911 |    0.7556 |    0.8156 | 0.5219 |
| ft384o64     | hybrid    |    96,134 |         1.0 |          1.0 |         100 |    20 |   0.4533 |   0.5667 |   0.6111 |    0.6844 |    0.7333 |    0.8089 | 0.5230 |



quality baseline сейчас - ft256o32
но ft384o64 сохраняет почти такое же качество, при этом содержит на 30% меньше chunks (cost-performance effective)




TOO MUCH
Loaded 28,481 documents
Building ha384o64m512r2v1...
Chunks: 342,624
Mean chunks/document: 12.03
Median chunks/document: 9.00
P95 chunks/document: 27
Mean body tokens/chunk: 72.04
P95 body tokens/chunk: 335
Chunks under 50 tokens: 213,524 (62.32%)
Chunks with section path: 327,161 (95.49%)
Unique section paths: 60,409
Maximum section depth: 4
Formatted chunks over budget: 0 (0.00%)
Output: /home/cohle/machine_learning/supportbench/data/nvidia_techqa/chunks/ha384o64m512r2v1



heading aware chunker v2 is much better
Loaded 28,481 documents
Building ha384o64m512r2v2...
Chunks: 165,623
Mean chunks/document: 5.82
Median chunks/document: 6.00
P95 chunks/document: 10
Mean body tokens/chunk: 169.76
P95 body tokens/chunk: 384
Chunks under 50 tokens: 45,811 (27.66%)
Chunks with section path: 121,514 (73.37%)
Unique section paths: 2,600
Maximum section depth: 3
Formatted chunks over budget: 0 (0.00%)


Общие параметры: split=train, queries=600, из них 450 labeled и 150 unlabeled

| Chunk config     | Retriever | Documents | BM25 weight | Dense weight | Candidate k | RRF k | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | MRR@10 |
| ---------------- | --------- | --------: | ----------: | -----------: | ----------: | ----: | -------: | -------: | -------: | --------: | --------: | --------: | -----: |
| ha384o64m512r2v2 | bm25      |   165,623 |           — |            — |           — |     — |   0.4222 |   0.5000 |   0.5511 |    0.6067 |    0.6622 |    0.7289 | 0.4765 |
| ha384o64m512r2v2 | dense     |   165,623 |           — |            — |           — |     — |   0.4200 |   0.5489 |   0.5889 |    0.6667 |    0.7289 |    0.7911 | 0.4958 |
| ha384o64m512r2v2 | hybrid    |   165,623 |         1.0 |          1.0 |         100 |    20 |   0.4689 |   0.5822 |   0.6133 |    0.6733 |    0.7556 |    0.8044 | 0.5344 |









| Chunk config     | Candidate pool | Indexed chunks | Candidate coverage | Before R@1 | Before R@3 | Before R@5 | Before R@10 | Before MRR | After R@1 | After R@3 | After R@5 | After R@10 | After MRR |    ΔR@1 |    ΔR@3 |    ΔR@5 |   ΔR@10 |    ΔMRR |
| ---------------- | -------------: | -------------: | -----------------: | ---------: | ---------: | ---------: | ----------: | ---------: | --------: | --------: | --------: | ---------: | --------: | ------: | ------: | ------: | ------: | ------: |
| ft256o32         |             20 |        135,235 |             0.7556 |     0.4467 |     0.5733 |     0.6178 |      0.6911 |     0.5219 |    0.5000 |    0.6022 |    0.6533 |     0.7178 |    0.5667 | +0.0533 | +0.0289 | +0.0356 | +0.0267 | +0.0448 |
| ft256o32         |             50 |        135,235 |             0.8156 |     0.4467 |     0.5733 |     0.6178 |      0.6911 |     0.5219 |    0.4778 |    0.5933 |    0.6422 |     0.7067 |    0.5486 | +0.0311 | +0.0200 | +0.0244 | +0.0156 | +0.0267 |
| ha384o64m512r2v2 |             20 |        165,623 |             0.7556 |     0.4689 |     0.5822 |     0.6133 |      0.6733 |     0.5344 |    0.4844 |    0.6022 |    0.6578 |     0.7156 |    0.5567 | +0.0156 | +0.0200 | +0.0444 | +0.0422 | +0.0223 |
| ha384o64m512r2v2 |             50 |        165,623 |             0.8044 |     0.4689 |     0.5822 |     0.6133 |      0.6733 |     0.5344 |    0.4667 |    0.5911 |    0.6467 |     0.7000 |    0.5402 | -0.0022 | +0.0089 | +0.0333 | +0.0267 | +0.0058 |


ft384o64:
efficiency baseline

heading-aware v2:
structural chunking baseline

pool 50:
higher candidate coverage but worse reranking


On dev split
Candidate coverage: 0.7812
Before reranking:
  R@1 : 0.4625
  R@3 : 0.6188
  R@5 : 0.6625
  R@10: 0.7188
  R@20: 0.7812
  MRR:  0.5508
After reranking:
  R@1 : 0.4688
  R@3 : 0.6125
  R@5 : 0.6875
  R@10: 0.7438
  MRR:  0.5590



## Current fixed configuration:
  - Parent WRRF + parent reranking/fusion.
  - parent_candidate_k=20.
  - candidate_prior_weight=1.25.
  - top_parents=4.
  - Within-parent cross-encoder selection.
  - chunks_per_parent=2.
  - max_context_tokens=4096.
  - model_context_window=8192.
  - reserved_output_tokens=1024.
  - Dense на CUDA.



# Enterprise system

PostgreSQL persistence
        │
        ├── isolated worlds
        ├── services
        ├── assets / installed products
        ├── users / entitlements
        ├── support cases
        └── audit events

EnterpriseService
        │
        ├── get_service_status
        ├── get_installed_product
        ├── check_user_entitlement
        └── create_support_case
                │
                ├── business assignment
                ├── Clock
                ├── atomic audit
                └── race-safe idempotency



## Security boundary
untrusted
ToolCall.arguments
       │
       ▼
 Pydantic extra=forbid
       │
       ▼
   Tool Handler
       │
       ▼
trusted context
world / actor / request
       │
       ▼
EnterpriseService


## End-to-end tool calling pipeline
ToolCall
  ↓
ToolGateway
  ↓
Pydantic validation
  ↓
enterprise handler
  ↓
EnterpriseService
  ↓
PostgresUnitOfWork
  ↓
PostgreSQL
